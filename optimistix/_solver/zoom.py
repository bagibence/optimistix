import functools as ft
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

from .._custom_types import Y
from .._misc import lin_to_grad, tree_dot, tree_full_like
from .._search import _FnEvalInfo, _FnInfo
from .._solution import RESULTS


IntScalar = Int[Scalar, ""]
FloatScalar = Float[Scalar, ""]


# from optax
def _cond_print(condition, message, **kwargs):
    """Prints message if condition is true."""
    jax.lax.cond(
        condition,
        lambda _: jax.debug.print(message, **kwargs, ordered=True),
        lambda _: None,
        None,
    )


def quadratic_min(a, val_a, grad_a, b, val_b):
    dist = b - a
    upper = -grad_a * dist**2
    lower = 2 * (val_b - val_a - grad_a * dist)
    return a + upper / lower


def optax_cubicmin(a, fa, fpa, b, fb, c, fc):
    """Cubic interpolation.

    Finds a critical point of a cubic polynomial
    p(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D, that goes through
    the points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    May return NaN (if radical<0), in that case, the point will be ignored.
    Adapted from scipy.optimize._linesearch.py.

    Args:
      a: scalar
      fa: value of a function f at a
      fpa: slope of a function f at a
      b: scalar
      fb: value of a function f at b
      c: scalar
      fc: value of a function f at c

    Returns:
      xmin: point at which p'(xmin) = 0
    """
    C = fpa
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
    A, B = (
        jnp.dot(
            d1,
            jnp.array([fb - fa - C * db, fc - fa - C * dc]),
            precision=jax.lax.Precision.HIGHEST,
        )
        / denom
    )

    radical = B * B - 3.0 * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

    return xmin


def interpolate(lo, value_lo, slope_lo, hi, value_hi, cubic_ref, value_cubic_ref):
    """
    Find
    """
    # Check if interval not too small otherwise fail
    delta = jnp.abs(hi - lo)
    left = jnp.minimum(hi, lo)
    right = jnp.maximum(hi, lo)
    cubic_chk = 0.2 * delta  # cubic guess has to be at least this far from the sides
    quad_chk = 0.1 * delta  # quadratic guess has to be at least this far from the sides

    # too_small_int = delta <= min_interval_length

    middle_cubic = optax_cubicmin(
        lo, value_lo, slope_lo, hi, value_hi, cubic_ref, value_cubic_ref
    )
    middle_cubic_valid = (middle_cubic > left + cubic_chk) & (
        middle_cubic < right - cubic_chk
    )
    middle_quad = quadratic_min(lo, value_lo, slope_lo, hi, value_hi)
    middle_quad_valid = (middle_quad > left + quad_chk) & (
        middle_quad < right - quad_chk
    )
    middle_bisection = (lo + hi) / 2.0

    a_j = middle_bisection
    a_j = jnp.where(middle_quad_valid, middle_quad, a_j)
    a_j = jnp.where(middle_cubic_valid, middle_cubic, a_j)

    return a_j


def decrease_condition(stepsize_i, value_a_i, value_init, slope_init, c1):
    return value_a_i <= value_init + c1 * stepsize_i * slope_init


def curvature_condition(slope_a_i, slope_init, c2):
    return jnp.abs(slope_a_i) <= c2 * jnp.abs(slope_init)


def tree_where(cond, candidate, default):
    def _set_val(x, y):
        return jnp.where(cond, x, y)

    return jax.tree.map(_set_val, candidate, default)


# copied from optax.tree_utils
def tree_add_scale(tree_x, scalar, tree_y):
    scalar = jnp.asarray(scalar)
    return jax.tree.map(
        lambda x, y: None if x is None else x + scalar.astype(x.dtype) * y,
        tree_x,
        tree_y,
        is_leaf=lambda x: x is None,
    )


def tree_sub(tree_x, tree_y):
    return jax.tree.map(operator.sub, tree_x, tree_y)


# like FunctionInfo.Eval
class PointEval(eqx.Module):
    location: jax.Array
    value: FloatScalar


# like FunctionInfo.EvalGrad
class PointEvalGrad(eqx.Module):
    location: jax.Array
    value: FloatScalar
    grad: jax.Array

    def compute_grad_dot(self, y):
        return tree_dot(self.grad, y)

    def strip_grad(self) -> PointEval:
        return PointEval(self.location, self.value)


class ZoomState(eqx.Module):
    ls_iter_num: IntScalar
    #
    init_stepsize_guess: FloatScalar
    init_point: PointEvalGrad
    slope_init: FloatScalar
    #
    stepsize: FloatScalar
    point_i: PointEvalGrad
    slope_i: FloatScalar
    #
    interval_found: Bool
    done: Bool
    failed: Bool
    #
    stepsize_lo: FloatScalar
    point_lo: PointEvalGrad
    slope_lo: FloatScalar
    stepsize_hi: FloatScalar
    point_hi: PointEval
    #
    cubic_ref_stepsize: FloatScalar
    cubic_ref_point: PointEval
    #
    safe_stepsize: FloatScalar
    safe_point: PointEvalGrad
    #
    proposed_stepsize: FloatScalar


class Zoom(eqx.Module):
    c1: float = 1e-4
    c2: float = 0.9
    max_stepsize: float = 1.0
    increase_factor: float = 1.5
    initial_guess_strategy: str = "keep"
    min_interval_length: float = 1e-6
    maxls: int = 30
    verbose: bool = False

    # TODO decide on defaults

    @staticmethod
    def _try_replace_stepsize_with_safe(state):
        # NOTE this doesn't replace the slope
        final_stepsize, final_point = tree_where(
            (state.safe_stepsize > 0.0),
            [state.safe_stepsize, state.safe_point],
            [state.stepsize, state.point_i],
        )
        state = eqx.tree_at(
            lambda s: (s.stepsize, s.point_i),
            state,
            (final_stepsize, final_point),
        )

        return state

    def _init_stepsize(
        self,
        prev_stepsize,
    ):
        match self.initial_guess_strategy:
            case "one":
                a_i = 1.0
            case "keep":
                a_i = prev_stepsize
            case "increase":
                a_i = prev_stepsize * self.increase_factor
            case _:
                raise ValueError(
                    "initial_guess_strategy has to one of ('one', 'keep', 'increase')"
                )

        a_i = jnp.where(a_i < self.min_interval_length, self.max_stepsize, a_i)
        a_i = jnp.minimum(a_i, self.max_stepsize)

        return jnp.array(a_i)

    def _actual_init(self, init_point, prev_ls_stepsize, descent_direction):
        # init_point is where stepsize = 0
        _slope_init = init_point.compute_grad_dot(descent_direction)

        init_stepsize = self._init_stepsize(prev_ls_stepsize)
        proposed_stepsize = init_stepsize

        return ZoomState(
            ls_iter_num=jnp.array(0),
            init_stepsize_guess=init_stepsize,
            init_point=init_point,
            slope_init=_slope_init,
            #
            stepsize=jnp.array(0.0),
            point_i=init_point,
            slope_i=_slope_init,
            #
            interval_found=jnp.array(False),
            done=jnp.array(False),
            failed=jnp.array(False),
            #
            stepsize_lo=jnp.array(0.0),
            point_lo=init_point,
            slope_lo=_slope_init,
            stepsize_hi=jnp.array(0.0),
            point_hi=init_point.strip_grad(),
            #
            cubic_ref_stepsize=jnp.array(0.0),
            cubic_ref_point=init_point.strip_grad(),
            #
            safe_stepsize=jnp.array(0.0),
            safe_point=init_point,
            #
            proposed_stepsize=proposed_stepsize,
        )

    def init(self, y, f_info_struct):
        if self.verbose:
            jax.debug.print("Doing empty init")

        # empty init
        # it's called once when the whole optimization starts
        _slope_init = jnp.array(-jnp.inf)
        init_point = PointEvalGrad(y, jnp.array(jnp.inf), tree_full_like(y, 0.0))
        init_stepsize = jnp.array(1.0)

        return ZoomState(
            ls_iter_num=jnp.array(0),
            init_stepsize_guess=init_stepsize,
            init_point=init_point,
            slope_init=_slope_init,
            #
            stepsize=jnp.array(0.0),
            point_i=init_point,
            slope_i=_slope_init,
            #
            interval_found=jnp.array(False),
            done=jnp.array(False),
            failed=jnp.array(False),
            #
            stepsize_lo=jnp.array(0.0),
            point_lo=init_point,
            slope_lo=_slope_init,
            stepsize_hi=jnp.array(0.0),
            point_hi=init_point.strip_grad(),
            #
            cubic_ref_stepsize=jnp.array(0.0),
            cubic_ref_point=init_point.strip_grad(),
            #
            safe_stepsize=jnp.array(0.0),
            safe_point=init_point,
            #
            proposed_stepsize=jnp.array(0.0),
        )

    def _propose_by_interpolation(self, state):
        stepsize_middle = interpolate(
            state.stepsize_lo,
            state.point_lo.value,
            state.slope_lo,
            state.stepsize_hi,
            state.point_hi.value,
            state.cubic_ref_stepsize,
            state.cubic_ref_point.value,
        )
        return stepsize_middle

    def _propose_by_increase(self, state):
        # propose a stepsize
        # on the first step of the current search use the one that was initialized using the previous search's stepsize
        #   which is one of: previous step's final stepsize; that increased by increased factor; 1.0
        # on all other iterations, increase by increase_factor
        stepsize_i = jnp.where(
            state.ls_iter_num == 0,
            state.init_stepsize_guess,
            state.stepsize * self.increase_factor,
        )
        # guard from above by max stepsize
        stepsize_i = jnp.minimum(stepsize_i, self.max_stepsize)
        reached_max_stepsize = stepsize_i >= self.max_stepsize

        # TODO reached_max_stepsize has to be documented in the state

        return stepsize_i

    def _propose_on_first_iter(self, state):
        return self._init_stepsize(state.stepsize)

    def propose_stepsize(self, state):
        return jax.lax.cond(
            state.interval_found,
            self._propose_by_interpolation,
            self._propose_by_increase,
            state,
        )

    def _zoom_into_interval(
        self, y, y_eval, f_info, f_eval_info, y_eval_grad, descent_direction, state
    ):
        stepsize_lo = state.stepsize_lo
        value_lo = state.point_lo.value
        slope_lo = state.slope_lo
        stepsize_hi = state.stepsize_hi
        value_hi = state.point_hi.value

        value_init = state.init_point.value

        if self.verbose:
            jax.debug.print("Zooming into interval: ({}, {})", stepsize_lo, stepsize_hi)

        # TODO is state.stepsize always the last accepted step?

        # y_eval was created by taking state.proposed_stepsize
        stepsize_middle = state.proposed_stepsize
        point_middle = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_middle = point_middle.compute_grad_dot(descent_direction)

        middle_satisf_decrease = decrease_condition(
            stepsize_middle, point_middle.value, value_init, state.slope_init, self.c1
        )
        middle_satisf_curvature = curvature_condition(
            slope_middle, state.slope_init, self.c2
        )

        if self.verbose:
            jax.debug.print(
                "Middle is: {}\tDecrease: {}\tCurv: {}",
                stepsize_middle,
                middle_satisf_decrease,
                middle_satisf_curvature,
            )

        # if it doesn't satisfy the decrease condition
        # or its value is worse than the previous best value that satisfies it
        middle_better_than_lo = point_middle.value < value_lo

        update_safe_stepsize = middle_satisf_decrease & (
            point_middle.value < state.safe_point.value
        )
        new_safe_stepsize, new_safe_point = tree_where(
            update_safe_stepsize,
            [stepsize_middle, point_middle],
            [state.safe_stepsize, state.safe_point],
        )

        #
        middle_slope_satisf_third_cond = slope_middle * (stepsize_hi - stepsize_lo) >= 0

        # new point is not better than lo, so replace the hi side with it, keep lo as lo
        set_hi_to_middle = (~middle_satisf_decrease) | (~middle_better_than_lo)

        # new point is better than lo, so it will be the new lo and lo will be the new hi
        set_hi_to_lo = ~set_hi_to_middle & middle_slope_satisf_third_cond

        # same as set_lo_to_middle = not set_hi_to_middle
        set_lo_to_middle = (
            middle_satisf_decrease & middle_better_than_lo
            # & ~middle_slope_satisf_third_cond
        )

        set_cubic_to_hi_1 = ~(middle_satisf_decrease & middle_better_than_lo)
        set_cubic_to_hi = set_cubic_to_hi_1 | middle_slope_satisf_third_cond
        # same as set_hi_to_middle or set_lo_to_middle?

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_middle,
            (stepsize_middle, point_middle.strip_grad()),
            (stepsize_hi, state.point_hi),
        )

        # update the reference point
        new_cubic_ref, new_cubic_ref_point = tree_where(
            set_cubic_to_hi,
            (stepsize_hi, state.point_hi),
            (stepsize_lo, PointEval(state.point_lo.location, state.point_lo.value)),
        )

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_lo,
            (stepsize_lo, PointEval(state.point_lo.location, state.point_lo.value)),
            (new_stepsize_hi, new_point_hi),
        )

        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            set_lo_to_middle,
            (stepsize_middle, point_middle, slope_middle),
            (stepsize_lo, state.point_lo, state.slope_lo),
        )

        # accept_middle = (
        #    middle_satisf_decrease & middle_satisf_curvature & middle_better_than_lo
        # )
        accept_middle = middle_satisf_decrease & middle_satisf_curvature

        interval_too_short = (
            jnp.abs(stepsize_hi - stepsize_lo) < self.min_interval_length
        )

        # done = interval_too_short or accept_middle
        done = accept_middle

        max_iter_reached = (state.ls_iter_num + 1) >= self.maxls
        presumably_failed = max_iter_reached | (
            interval_too_short & (new_safe_stepsize > 0.0)
        )
        failed = presumably_failed & (~done)

        if self.verbose:
            _cond_print(
                interval_too_short,
                "Interval too short: ({ss_lo}, {ss_hi})",
                ss_lo=stepsize_lo,
                ss_hi=stepsize_hi,
            )

        return ZoomState(
            ls_iter_num=state.ls_iter_num + 1,
            #
            init_stepsize_guess=state.init_stepsize_guess,
            init_point=state.init_point,
            slope_init=state.slope_init,
            #
            stepsize=stepsize_middle,
            point_i=point_middle,
            slope_i=slope_middle,
            #
            stepsize_lo=new_stepsize_lo,
            point_lo=new_point_lo,
            slope_lo=new_slope_lo,
            stepsize_hi=new_stepsize_hi,
            point_hi=new_point_hi,
            #
            interval_found=state.interval_found,
            done=done,
            failed=failed,
            #
            cubic_ref_stepsize=new_cubic_ref,
            cubic_ref_point=new_cubic_ref_point,
            #
            safe_stepsize=new_safe_stepsize,
            safe_point=new_safe_point,
            #
            proposed_stepsize=state.proposed_stepsize,
        )

    def _search_interval(
        self, y, y_eval, f_info, f_eval_info, y_eval_grad, descent_direction, state
    ):
        # evaluate the slope along the descent direction for the new stepsize
        stepsize_i = state.proposed_stepsize
        point_i = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_a_i = point_i.compute_grad_dot(descent_direction)

        reached_max_stepsize = stepsize_i >= self.max_stepsize

        # Check the conditions for the new point
        decrease_satisfied = decrease_condition(
            stepsize_i,
            point_i.value,
            state.init_point.value,
            state.slope_init,
            self.c1,
        )
        value_increased = point_i.value >= state.point_i.value
        curvature_satisfied = curvature_condition(slope_a_i, state.slope_init, self.c2)

        # save this as the largest stepsize that satisfies the decrease condition
        new_safe_stepsize, new_safe_point = tree_where(
            decrease_satisfied,
            [stepsize_i, point_i],
            [state.safe_stepsize, state.safe_point],
        )

        # There are two conditions when we say we found an interval and call _zoom_into_interval
        found_a = (~decrease_satisfied) | (value_increased & state.ls_iter_num > 0)
        found_b = (~found_a) & (~curvature_satisfied) & (slope_a_i >= 0)
        interval_found = found_a | found_b
        # assert ~(found_a & found_b)

        # optax calls ^these^ set_high_to_new and set_low_to_new
        # set_high_to_new = (decrease_error > 0.0) | ((new_value_step >= prev_value_step) & (iter_num > 0))
        # set_low_to_new = (new_slope_step >= 0.0) & (~set_high_to_new)

        a_prev_is_lo = found_a

        # state.stepsize and point_i are the same as prev_stepsize earlier
        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            a_prev_is_lo,
            (state.stepsize, state.point_i, state.slope_i),
            (stepsize_i, point_i, slope_a_i),
        )
        new_stepsize_hi, new_point_hi = tree_where(
            a_prev_is_lo,
            (stepsize_i, point_i.strip_grad()),
            (state.stepsize, state.point_i.strip_grad()),
        )

        if self.verbose:
            _cond_print(
                decrease_satisfied, "Decrease satisfied for {ss}", ss=stepsize_i
            )

        if self.verbose:
            _cond_print(
                curvature_satisfied, "Curvature satisfied for {ss}", ss=stepsize_i
            )

        # from optax
        done = (decrease_satisfied & curvature_satisfied) | (
            reached_max_stepsize & ~interval_found
        )
        failed = (state.ls_iter_num + 1 >= self.maxls) & (~done)

        return ZoomState(
            ls_iter_num=state.ls_iter_num + 1,
            #
            init_stepsize_guess=state.init_stepsize_guess,
            init_point=state.init_point,
            slope_init=state.slope_init,
            #
            stepsize=stepsize_i,
            point_i=point_i,
            slope_i=slope_a_i,
            #
            stepsize_lo=new_stepsize_lo,
            point_lo=new_point_lo,
            slope_lo=new_slope_lo,
            stepsize_hi=new_stepsize_hi,
            point_hi=new_point_hi,
            #
            cubic_ref_stepsize=new_stepsize_lo,
            cubic_ref_point=new_point_lo.strip_grad(),
            #
            interval_found=interval_found,
            done=done,
            failed=failed,
            #
            safe_stepsize=new_safe_stepsize,
            safe_point=new_safe_point,
            #
            proposed_stepsize=state.proposed_stepsize,
        )

    def do_first_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        lin_fn,
        options,
        state: ZoomState,
    ):
        proposed_stepsize = self._init_stepsize(state.stepsize)

        return proposed_stepsize, state

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        lin_fn,
        options,
        state: ZoomState,
    ):
        if self.verbose:
            jax.debug.print("state.ls_iter_num: {}", state.ls_iter_num)

        _first_step_fn = ft.partial(
            self.do_first_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )
        _regular_step_fn = ft.partial(
            self.do_regular_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )

        # TODO Condition could also be something like something in the state is inf
        proposed_stepsize, state = jax.lax.cond(
            first_step,
            _first_step_fn,
            _regular_step_fn,
            state,
        )

        # accept = first_step | state.done
        accept = first_step | state.done | state.failed

        # set ls_iter_num back to 0 if accepted
        new_ls_iter_num = jnp.where(accept, jnp.array(0), state.ls_iter_num)
        state = eqx.tree_at(lambda s: s.ls_iter_num, state, new_ls_iter_num)

        if self.verbose:
            jax.debug.print("Proposing {}", proposed_stepsize)

        return (proposed_stepsize, accept, RESULTS.successful, state)

    def do_regular_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        lin_fn,
        options,
        state: ZoomState,
    ):
        if self.verbose:
            jax.debug.print("Linesearch regular iter: {}", state.ls_iter_num)

        y_eval_grad = lin_to_grad(
            lin_fn, y_eval, autodiff_mode=options.get("autodiff_mode", "bwd")
        )
        descent_direction = tree_sub(y_eval, y)

        reinit_state = self._actual_init(
            PointEvalGrad(y, f_info.f, f_info.grad),
            state.stepsize,  # at this point it should be the last stepsize we took in the last linesearch
            descent_direction,
        )
        state = jax.lax.cond(
            state.ls_iter_num == 0,
            lambda: reinit_state,
            lambda: state,
        )

        _zoom_fn = ft.partial(
            self._zoom_into_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
            y_eval_grad,
            descent_direction,
        )
        _search_fn = ft.partial(
            self._search_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
            y_eval_grad,
            descent_direction,
        )

        state = jax.lax.cond(
            state.interval_found,
            _zoom_fn,
            _search_fn,
            state,
        )

        # if failed, try the safe stepsize instead
        state = jax.lax.cond(
            state.failed,
            Zoom._try_replace_stepsize_with_safe,
            lambda state: state,
            state,
        )

        if self.verbose:
            jax.debug.print("Checked {}. Done: {}", state.stepsize, state.done)

        proposed_stepsize = self.propose_stepsize(state)
        state = eqx.tree_at(lambda s: s.proposed_stepsize, state, proposed_stepsize)

        return proposed_stepsize, state

import functools as ft
import operator

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar

from .._custom_types import Y
from .._misc import lin_to_grad, tree_dot, tree_full_like
from .._search import _FnEvalInfo, _FnInfo, AbstractSearch
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
    # adapted from optax
    delta = jnp.abs(hi - lo)
    left = jnp.minimum(hi, lo)
    right = jnp.maximum(hi, lo)
    cubic_chk = 0.2 * delta  # cubic guess has to be at least this far from the sides
    quad_chk = 0.1 * delta  # quadratic guess has to be at least this far from the sides

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

    jax.debug.print(
        "lo {}, value_lo {}, slope_lo {}, hi {}, value_hi {}, cubic_ref {}, value_cubic_ref {}",
        lo,
        value_lo,
        slope_lo,
        hi,
        value_hi,
        cubic_ref,
        value_cubic_ref,
    )
    jax.debug.print("{}, {}", middle_cubic, middle_quad)

    return a_j


def decrease_condition_with_approx(
    stepsize, value_step, slope_step, value_init, slope_init, c1, c3
):
    # adopted from jaxopt and optax
    decrease_error = value_step - value_init - c1 * stepsize * slope_init
    if c3 is not None:
        approx_decrease_error = slope_step - (2 * c1 - 1.0) * slope_init

        delta_values = value_step - value_init - c3 * jnp.abs(value_init)
        approx_decrease_error = jnp.maximum(approx_decrease_error, delta_values)
        decrease_error = jnp.minimum(approx_decrease_error, decrease_error)

    return decrease_error <= 0.0


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
    current_point: PointEvalGrad
    current_slope: FloatScalar
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
    y_eval_stepsize: FloatScalar
    #
    descent_direction: jax.Array


class Zoom(AbstractSearch[Y, _FnInfo, _FnEvalInfo, ZoomState], strict=True):
    c1: float = 1e-4
    c2: float = 0.9
    c3: float = 1e-6
    max_stepsize: float = 1.0
    increase_factor: float = 1.5
    initial_guess_strategy: str = "keep"
    min_interval_length: float = 1e-6
    min_stepsize: float = 1e-6
    maxls: int = 30
    verbose: bool = False

    # TODO decide on defaults

    @staticmethod
    def _try_replace_stepsize_with_safe(state):
        # NOTE this doesn't replace the slope
        final_stepsize, final_point = tree_where(
            (state.safe_stepsize > 0.0),
            [state.safe_stepsize, state.safe_point],
            [state.stepsize, state.current_point],
        )
        final_slope = final_point.compute_grad_dot(state.descent_direction)
        state = eqx.tree_at(
            lambda s: (s.stepsize, s.current_point, s.current_slope),
            state,
            (final_stepsize, final_point, final_slope),
        )

        return state

    def init_stepsize_from_previous(
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

        # reset if too small
        a_i = jnp.where(a_i <= self.min_stepsize, self.max_stepsize, a_i)
        # guard from above by max_stepsize
        a_i = jnp.minimum(a_i, self.max_stepsize)

        return jnp.array(a_i)

    def _actual_init(self, init_point, prev_ls_stepsize, y_eval):
        # init_point is where stepsize = 0
        # currently y_eval is with a stepsize of 1.0
        descent_direction = tree_sub(y_eval, init_point.location)
        _slope_init = init_point.compute_grad_dot(descent_direction)

        init_stepsize = self.init_stepsize_from_previous(prev_ls_stepsize)
        proposed_stepsize = init_stepsize

        return ZoomState(
            ls_iter_num=jnp.array(0),
            init_stepsize_guess=init_stepsize,
            init_point=init_point,
            slope_init=_slope_init,
            #
            stepsize=jnp.array(0.0),
            current_point=init_point,
            current_slope=_slope_init,
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
            y_eval_stepsize=proposed_stepsize,
            #
            descent_direction=descent_direction,
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
            current_point=init_point,
            current_slope=_slope_init,
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
            y_eval_stepsize=jnp.array(0.0),
            #
            descent_direction=tree_full_like(y, 0.0),
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
        new_stepsize = jnp.where(
            state.ls_iter_num == 0,
            state.init_stepsize_guess,
            state.stepsize * self.increase_factor,
        )
        # guard from above by max stepsize
        new_stepsize = jnp.minimum(new_stepsize, self.max_stepsize)

        return new_stepsize

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

        # y_eval was created by taking state.y_eval_stepsize
        stepsize_middle = state.y_eval_stepsize
        point_middle = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_middle = point_middle.compute_grad_dot(descent_direction)
        jax.debug.print("descent_direction for slope_middle: {}", descent_direction)
        jax.debug.print("slope_middle: {}", slope_middle)

        # middle_satisf_decrease = decrease_condition(
        #    stepsize_middle, point_middle.value, value_init, state.slope_init, self.c1
        # )
        middle_satisf_decrease = decrease_condition_with_approx(
            stepsize_middle,
            point_middle.value,
            slope_middle,
            value_init,
            state.slope_init,
            self.c1,
            self.c3,
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

        # TODO decide which one to use: largest step or best function value
        # update_safe_stepsize = middle_satisf_decrease & (
        #    point_middle.value < state.safe_point.value
        # )
        update_safe_stepsize = middle_satisf_decrease & (
            stepsize_middle > state.safe_stepsize
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

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_middle,
            (stepsize_middle, point_middle.strip_grad()),
            (stepsize_hi, state.point_hi),
        )

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_lo,
            (stepsize_lo, state.point_lo.strip_grad()),
            (new_stepsize_hi, new_point_hi),
        )

        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            set_lo_to_middle,
            (stepsize_middle, point_middle, slope_middle),
            (stepsize_lo, state.point_lo, state.slope_lo),
        )

        # update the reference point
        # set_cubic_to_hi_1 = ~(middle_satisf_decrease & middle_better_than_lo)
        # set_cubic_to_hi = set_cubic_to_hi_1 | middle_slope_satisf_third_cond
        set_cubic_to_hi = set_hi_to_middle | set_hi_to_lo
        new_cubic_ref, new_cubic_ref_point = tree_where(
            set_cubic_to_hi,
            (stepsize_hi, state.point_hi),
            (stepsize_lo, state.point_lo.strip_grad()),
        )

        # accept_middle = (
        #    middle_satisf_decrease & middle_satisf_curvature & middle_better_than_lo
        # )
        accept_middle = middle_satisf_decrease & middle_satisf_curvature

        interval_too_short = (
            # jnp.abs(stepsize_hi - stepsize_lo) <= self.min_interval_length
            jnp.abs(new_stepsize_hi - new_stepsize_lo) <= self.min_interval_length
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
            current_point=point_middle,
            current_slope=slope_middle,
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
            y_eval_stepsize=state.y_eval_stepsize,
            #
            descent_direction=state.descent_direction,
        )

    def _search_interval(
        self, y, y_eval, f_info, f_eval_info, y_eval_grad, descent_direction, state
    ):
        # evaluate the slope along the descent direction for the new stepsize
        new_stepsize = state.y_eval_stepsize
        new_point = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_at_new_point = new_point.compute_grad_dot(descent_direction)
        jax.debug.print("descent_direction in _search_interval: {}", descent_direction)

        reached_max_stepsize = new_stepsize >= self.max_stepsize

        # Check the conditions for the new point
        # decrease_satisfied = decrease_condition(
        #    new_stepsize,
        #    new_point.value,
        #    state.init_point.value,
        #    state.slope_init,
        #    self.c1,
        # )
        decrease_satisfied = decrease_condition_with_approx(
            new_stepsize,
            new_point.value,
            slope_at_new_point,
            state.init_point.value,
            state.slope_init,
            self.c1,
            self.c3,
        )
        value_increased = new_point.value >= state.current_point.value
        curvature_satisfied = curvature_condition(
            slope_at_new_point, state.slope_init, self.c2
        )

        # save this as the largest stepsize that satisfies the decrease condition
        new_safe_stepsize, new_safe_point = tree_where(
            decrease_satisfied,
            [new_stepsize, new_point],
            [state.safe_stepsize, state.safe_point],
        )

        # There are two conditions when we say we found an interval and call _zoom_into_interval
        found_a = (~decrease_satisfied) | (value_increased & state.ls_iter_num > 0)
        found_b = (~found_a) & (~curvature_satisfied) & (slope_at_new_point >= 0)
        interval_found = found_a | found_b
        # assert ~(found_a & found_b)

        # if found_a: we call zoom(alpha_lo = alpha_{i-1}, alpha_hi = alpha_{i})
        # if found_b: we call zoom(alpha_lo = alpha_{i}, alpha_hi = alpha_{i-1})

        # state.stepsize is alpha_{i-1}
        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            found_a,
            (state.stepsize, state.current_point, state.current_slope),
            (new_stepsize, new_point, slope_at_new_point),
        )
        new_stepsize_hi, new_point_hi = tree_where(
            found_a,
            (new_stepsize, new_point.strip_grad()),
            (state.stepsize, state.current_point.strip_grad()),
        )

        if self.verbose:
            _cond_print(
                decrease_satisfied, "Decrease satisfied for {ss}", ss=new_stepsize
            )

        if self.verbose:
            _cond_print(
                curvature_satisfied, "Curvature satisfied for {ss}", ss=new_stepsize
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
            stepsize=new_stepsize,
            current_point=new_point,
            current_slope=slope_at_new_point,
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
            y_eval_stepsize=state.y_eval_stepsize,
            #
            descent_direction=state.descent_direction,
        )

    def fake_first_step(
        self,
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        lin_fn,
        options,
        state: ZoomState,
    ):
        # proposed_stepsize = self.init_stepsize_from_previous(state.stepsize)
        # state = eqx.tree_at(lambda s: s.y_eval_stepsize, state, proposed_stepsize)

        # or just propose 1.
        proposed_stepsize = jnp.array(1.0)

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

        _fake_first_step_fn = ft.partial(
            self.fake_first_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )
        _regular_step_fn = ft.partial(
            self.do_regular_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )

        # TODO Condition could also be something like something in the state is inf
        proposed_stepsize, state = jax.lax.cond(
            first_step,
            _fake_first_step_fn,
            _regular_step_fn,
            state,
        )

        # accept = first_step | state.done
        accept = first_step | state.done | state.failed

        # set ls_iter_num back to 0 if accepted
        new_ls_iter_num = jnp.where(accept, jnp.array(0), state.ls_iter_num)

        state = eqx.tree_at(
            lambda s: (s.ls_iter_num, s.y_eval_stepsize),
            state,
            (new_ls_iter_num, proposed_stepsize),
        )

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
        # jax.debug.print("Gradient eval")
        # descent_direction = tree_sub(y_eval, y)

        # def _reinit_state_fn():
        #    return self._actual_init(
        #        PointEvalGrad(y, f_info.f, f_info.grad),
        #        state.stepsize,  # at this point it should be the last stepsize we took in the last linesearch
        #        descent_direction,
        #    )

        _reinit_state_fn = ft.partial(
            self._actual_init,
            PointEvalGrad(y, f_info.f, f_info.grad),
            state.stepsize,  # at this point it should be the last stepsize we took in the last linesearch
            y_eval,
        )

        state = jax.lax.cond(
            state.ls_iter_num == 0,
            _reinit_state_fn,
            lambda: state,
        )

        _zoom_fn = ft.partial(
            self._zoom_into_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
            y_eval_grad,
            state.descent_direction,
        )
        _search_fn = ft.partial(
            self._search_interval,
            y,
            y_eval,
            f_info,
            f_eval_info,
            y_eval_grad,
            state.descent_direction,
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
        state = eqx.tree_at(lambda s: s.y_eval_stepsize, state, proposed_stepsize)

        return proposed_stepsize, state

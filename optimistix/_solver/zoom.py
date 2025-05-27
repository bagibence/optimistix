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


def _cond_print(condition, message, **kwargs):
    """Prints message if condition is true. From Optax."""
    jax.lax.cond(
        condition,
        lambda _: jax.debug.print(message, **kwargs, ordered=True),
        lambda _: None,
        None,
    )


def quadratic_min(
    a: float, value_a: float, slope_a: float, b: float, value_b: float
) -> float:
    """
    Quadratic interpolation.

    Find the minimum of the curve fitted to (a, value_a) and (b, value_b)
    """
    dist = b - a
    upper = -slope_a * dist**2
    lower = 2 * (value_b - value_a - slope_a * dist)
    return a + upper / lower


def optax_cubicmin(
    a: FloatScalar,
    value_a: FloatScalar,
    slope_a: FloatScalar,
    b: FloatScalar,
    value_b: FloatScalar,
    c: FloatScalar,
    value_c: FloatScalar,
) -> FloatScalar:
    """
    Cubic interpolation. Adapted from Optax.

    Finds a critical point of a cubic polynomial
    p(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D, that goes through
    the points (a,value_a), (b,value_b), and (c,value_c) with derivative at a of slope_a.
    May return NaN (if radical<0), in that case, the point will be ignored.
    Adapted from scipy.optimize._linesearch.py.

    Args:
      a: scalar
      value_a: value of a function f at a
      slope_a: slope of a function f at a
      b: scalar
      value_b: value of a function f at b
      c: scalar
      value_c: value of a function f at c

    Returns:
      xmin: point at which p'(xmin) = 0
    """
    C = slope_a
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -(db**2)], [-(dc**3), db**3]])
    A, B = (
        jnp.dot(
            d1,
            jnp.array([value_b - value_a - C * db, value_c - value_a - C * dc]),
            precision=jax.lax.Precision.HIGHEST,
        )
        / denom
    )

    radical = B * B - 3.0 * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3.0 * A)

    return xmin


def interpolate(
    lo: FloatScalar,
    value_lo: FloatScalar,
    slope_lo: FloatScalar,
    hi: FloatScalar,
    value_hi: FloatScalar,
    cubic_ref: FloatScalar,
    value_cubic_ref: FloatScalar,
) -> FloatScalar:
    """
    Find a stepsize by minimizing the cubic or quadratic curve fitted to
    `lo`, `hi`, and `cubic_ref`.
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

    # jax.debug.print("interpolating")
    # jax.debug.print(
    #    "lo {}, value_lo {}, slope_lo {}, hi {}, value_hi {}, cubic_ref {}, value_cubic_ref {}",
    #    lo,
    #    value_lo,
    #    slope_lo,
    #    hi,
    #    value_hi,
    #    cubic_ref,
    #    value_cubic_ref,
    # )
    # jax.debug.print("{}, {}", middle_cubic, middle_quad)

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


def decrease_condition(
    stepsize_i: FloatScalar,
    value_a_i: FloatScalar,
    value_init: FloatScalar,
    slope_init: FloatScalar,
    c1: float,
) -> Bool:
    """
    Check whether stepsize_i satisfies the Armijo decrease condition.
    """
    return value_a_i <= value_init + c1 * stepsize_i * slope_init


def curvature_condition(slope_a_i: FloatScalar, slope_init: FloatScalar, c2: float):
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
    def _try_replace_stepsize_with_safe(state: ZoomState):
        """
        If the search failed, replace the stepsize stored in `state` with
        the safe stepsize.
        """
        jax.debug.print("Trying safe stepsize: {}", state.safe_stepsize)

        final_stepsize, final_point = tree_where(
            (state.safe_stepsize > 0.0),
            [state.safe_stepsize, state.safe_point],
            [state.stepsize, state.current_point],
        )
        # TODO why not just store this in the state as safe_slope?
        # recalculate the final slope for consistency
        final_slope = final_point.compute_grad_dot(state.descent_direction)
        state = eqx.tree_at(
            lambda s: (s.stepsize, s.current_point, s.current_slope),
            state,
            (final_stepsize, final_point, final_slope),
        )

        return state

    def init_stepsize_from_previous(self, prev_stepsize: FloatScalar) -> FloatScalar:
        """
        Initialize the linesearch's stepsize based on the previous steps size
        according to one of three strategies:
            - "one": initialize to 1.0. Recommended for quasi-Newton methods.
            - "keep": initialize to and start from the previous stepsize.
            - "increase": increase the previous stepsize by `increase_factor`.

        If the initial stepsize would be smaller than the smallest allowed stepsize
        (`min_stepsize`), reset it to `max_stepsize`.

        If the initial stepsize would be larger than the largest allowed stepsize
        (`max_stepsize`), clip it to `max_stepsize`.
        """
        match self.initial_guess_strategy:
            case "one":
                a_i = jnp.array(1.0)
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

        return a_i

    def _actual_init(
        self,
        init_point: PointEvalGrad,
        y_eval_stepsize: FloatScalar,
        y_eval: PointEval,
    ) -> ZoomState:
        # init_point is where stepsize = 0
        # currently y_eval is with a stepsize of 1.0
        descent_direction = tree_sub(y_eval, init_point.location)
        _slope_init = init_point.compute_grad_dot(descent_direction)

        # instead of initializing here, we use the stepsize that was proposed at the end
        # of the last linesearch step and was used to create the y_eval here
        init_stepsize = y_eval_stepsize

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
            y_eval_stepsize=y_eval_stepsize,
            #
            descent_direction=descent_direction,
        )

    def init(self, y, f_info_struct) -> ZoomState:
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
            y_eval_stepsize=jnp.array(-jnp.inf),
            #
            descent_direction=tree_full_like(y, 0.0),
        )

    def _propose_by_interpolation(self, state: ZoomState) -> FloatScalar:
        """
        Propose a stepsize by interpolation, fitting a curve to lo, hi, cubic_ref.
        """
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

    def _propose_by_increase(self, state: ZoomState) -> FloatScalar:
        """
        Propose a new stepsize by increasing the current one by `increase_factor`.
        """
        return state.stepsize * self.increase_factor

    def _interpolate_or_increase(self, state: ZoomState) -> FloatScalar:
        """
        Propose a stepsize in a way that depends on which stage of the zoom linesearch
        we are at.
        If the interval is found and we are zooming into it: interpolate.
        If the interval is not found yet: increase.
        """
        return jax.lax.cond(
            state.interval_found,
            self._propose_by_interpolation,
            self._propose_by_increase,
            state,
        )

    def propose_stepsize(self, state: ZoomState) -> FloatScalar:
        """
        Propose a stepsize to evaluate.

        On the first step of the current search use the initial stepsize guess which was
        made based on the previous search's final stepsize.
        See `Zoom.init_stepsize_from_previous`

        On all other iterations, propose based on the current stepsize and in which
        stage of tha zoom algorithm we are.

        In all cases, limit from above by the maximum allowed stepsize (`max_stepsize`).
        """
        # new_stepsize = jax.lax.cond(
        #    state.ls_iter_num == 0,
        #    lambda state: state.init_stepsize_guess,
        #    self._interpolate_or_increase,
        #    state,
        # )
        new_stepsize = self._interpolate_or_increase(state)

        # guard from above by max stepsize
        new_stepsize = jnp.minimum(new_stepsize, self.max_stepsize)

        return new_stepsize

    def _zoom_into_interval(
        self, y, y_eval, f_info, f_eval_info, y_eval_grad, descent_direction, state
    ):
        if self.verbose:
            jax.debug.print(
                "Zooming into interval: ({}, {})", state.stepsize_lo, state.stepsize_hi
            )

        # y_eval was created by taking state.y_eval_stepsize
        stepsize_middle = state.y_eval_stepsize
        point_middle = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_middle = point_middle.compute_grad_dot(descent_direction)
        # jax.debug.print("descent_direction for slope_middle: {}", descent_direction)
        # jax.debug.print("grad for slope_middle: {}", point_middle.grad)
        # jax.debug.print("slope_middle: {}", slope_middle)

        # check conditions for the middle point
        middle_satisf_decrease = decrease_condition_with_approx(
            stepsize_middle,
            point_middle.value,
            slope_middle,
            state.init_point.value,
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

        middle_lower_than_lo = point_middle.value < state.point_lo.value

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
        middle_slope_satisf_third_cond = (
            slope_middle * (state.stepsize_hi - state.stepsize_lo) >= 0
        )

        # new point is not better than lo, so replace the hi side with it, keep lo as lo
        set_hi_to_middle = (~middle_satisf_decrease) | (~middle_lower_than_lo)

        # new point is better than lo, so it will be the new lo
        set_lo_to_middle = middle_satisf_decrease & middle_lower_than_lo
        # same as set_lo_to_middle = not set_hi_to_middle

        # if we overwrite lo with the new point, then we
        # decide which side of the interval to keep based on the third condition
        # if the third condition is satisfied, lo is the new hi
        # otherwise hi stays hi
        set_hi_to_lo = set_lo_to_middle & middle_slope_satisf_third_cond

        # if set_hi_to_lo or set_hi_to_middle, then we overwrite hi
        # and can use it as the reference point
        # otherwise we changed lo, so keep that as reference
        set_cubic_to_hi = set_hi_to_middle | set_hi_to_lo

        # do the updates
        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_middle,
            (stepsize_middle, point_middle.strip_grad()),
            (state.stepsize_hi, state.point_hi),
        )

        new_stepsize_hi, new_point_hi = tree_where(
            set_hi_to_lo,
            (state.stepsize_lo, state.point_lo.strip_grad()),
            (new_stepsize_hi, new_point_hi),
        )

        new_stepsize_lo, new_point_lo, new_slope_lo = tree_where(
            set_lo_to_middle,
            (stepsize_middle, point_middle, slope_middle),
            (state.stepsize_lo, state.point_lo, state.slope_lo),
        )

        new_cubic_ref, new_cubic_ref_point = tree_where(
            set_cubic_to_hi,
            (state.stepsize_hi, state.point_hi),
            (state.stepsize_lo, state.point_lo.strip_grad()),
        )

        # if middle satisfies both conditions, then we accept it as the final stepsize
        done = middle_satisf_decrease & middle_satisf_curvature

        interval_too_short = (
            jnp.abs(new_stepsize_hi - new_stepsize_lo) <= self.min_interval_length
        )

        # diagnose failure the same way optax does
        max_iter_reached = (state.ls_iter_num + 1) >= self.maxls
        presumably_failed = max_iter_reached | (
            interval_too_short & (new_safe_stepsize > 0.0)
        )
        failed = presumably_failed & (~done)

        if self.verbose:
            _cond_print(
                interval_too_short,
                "Interval too short: ({ss_lo}, {ss_hi})",
                ss_lo=new_stepsize_lo,
                ss_hi=new_stepsize_hi,
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
        self,
        y: Y,
        y_eval: Y,
        f_info,
        f_eval_info,
        y_eval_grad,
        descent_direction: Y,
        state,
    ):
        # evaluate the slope along the descent direction for the new stepsize
        new_stepsize = state.y_eval_stepsize
        new_point = PointEvalGrad(y_eval, f_eval_info.f, y_eval_grad)
        slope_at_new_point = new_point.compute_grad_dot(descent_direction)

        # jax.debug.print("params_init in _search_interval: {}", y)
        # jax.debug.print("y_eval in _search_interval: {}", y_eval)
        # jax.debug.print("y_eval's value in _search_interval: {}", new_point.value)
        # jax.debug.print("descent_direction in _search_interval: {}", descent_direction)
        # jax.debug.print("grad_at_new_point: {}", new_point.grad)
        # jax.debug.print("slope_at_new_point: {}", slope_at_new_point)

        reached_max_stepsize = new_stepsize >= self.max_stepsize

        # Check the conditions for the new point
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
            (new_stepsize, new_point),
            (state.safe_stepsize, state.safe_point),
        )

        # There are two conditions when we say we found an interval
        found_a = (~decrease_satisfied) | (value_increased & state.ls_iter_num > 0)
        found_b = (~found_a) & (~curvature_satisfied) & (slope_at_new_point >= 0)
        interval_found = found_a | found_b

        # If the interval is found, from the next iteration on we do _zoom_into_interval
        # if found_a: we call zoom(alpha_lo = alpha_{i-1}, alpha_hi = alpha_{i})
        # if found_b: we call zoom(alpha_lo = alpha_{i}, alpha_hi = alpha_{i-1})
        # where state.stepsize is alpha_{i-1} and new_stepsize is alpha_{i}

        # If the interval is found, this will zoom into the correct interval.
        # If not, it still sets lo and hi, but that's okay because it will not be used
        # by _zoom_into_interval, and we will just return here in the next iteration.
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

        # TODO shouldn't we update the reference point?

        # if self.verbose:
        #    _cond_print(
        #        decrease_satisfied, "Decrease satisfied for {ss}", ss=new_stepsize
        #    )

        # if self.verbose:
        #    _cond_print(
        #        curvature_satisfied, "Curvature satisfied for {ss}", ss=new_stepsize
        #    )

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
        """
        Do nothing, just propose 1.0 on the very first step of the whole optimization.
        """
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
        _fake_first_step_fn = ft.partial(
            self.fake_first_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )
        _regular_step_fn = ft.partial(
            self.do_regular_step, y, y_eval, f_info, f_eval_info, lin_fn, options
        )

        proposed_stepsize, state = jax.lax.cond(
            first_step,
            _fake_first_step_fn,
            _regular_step_fn,
            state,
        )

        accept = first_step | state.done | state.failed

        # set ls_iter_num back to 0 if accepted
        new_ls_iter_num = jnp.where(accept, jnp.array(0), state.ls_iter_num)
        # and propose an initial stepsize for the next linesearch
        proposed_stepsize = jnp.where(
            accept,
            self.init_stepsize_from_previous(state.stepsize),
            proposed_stepsize,
        )

        _cond_print(accept, "Accepting {ss}", ss=state.stepsize)

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
        options: dict,
        state: ZoomState,
    ):
        if self.verbose:
            jax.debug.print("Linesearch regular iter: {}", state.ls_iter_num)

        y_eval_grad = lin_to_grad(
            lin_fn, y_eval, autodiff_mode=options.get("autodiff_mode", "bwd")
        )

        # on the first real iteration of the linesearch, reinitialize the state
        _reinit_state_fn = ft.partial(
            self._actual_init,
            PointEvalGrad(y, f_info.f, f_info.grad),
            state.y_eval_stepsize,  # proposed at the end of the previous linesearch
            y_eval,  # created with state.y_eval_stepsize
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

        # if failed, use the safe stepsize as the final one
        state = jax.lax.cond(
            state.failed,
            Zoom._try_replace_stepsize_with_safe,
            lambda state: state,
            state,
        )

        if self.verbose:
            jax.debug.print(
                "Checked {}. Done: {}. Failed: {}",
                state.stepsize,
                state.done,
                state.failed,
            )

        proposed_stepsize = jax.lax.cond(
            state.failed,
            lambda state: state.stepsize,
            self.propose_stepsize,
            state,
        )
        # proposed_stepsize = self.propose_stepsize(state)

        return proposed_stepsize, state

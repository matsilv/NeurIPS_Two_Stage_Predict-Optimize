# ip_model_whole(cGvector_vectorPenalty).py compute the gradient of c and G (rationalKS, G is a vector in this file)

import numpy as np
import scipy as sp
import time
import scipy.sparse as sps
import numbers
from warnings import warn
from scipy.linalg import LinAlgError
import torch
from torch import nn, optim
from torch.autograd import Variable,Function
import sys
import logging
import time, datetime
from scipy.optimize import OptimizeWarning
from remove_redundancy import _remove_redundancy, _remove_redundancy_sparse, _remove_redundancy_dense
np.set_printoptions(threshold=np.inf)
import gurobipy as gp
from gurobipy import GRB

#purchase_fee = 0.2
#compensation_fee = 0.25
varNum = 50
x_s1 = torch.zeros(varNum,dtype=torch.float64)

def _format_A_constraints(A, n_x, sparse_lhs=False):
    """Format the left hand side of the constraints to a 2D array

    Parameters
    ----------
    A : 2D array
        2D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.
    n_x : int
        The number of variables in the linear programming problem.
    sparse_lhs : bool
        Whether either of `A_ub` or `A_eq` are sparse. If true return a
        coo_matrix instead of a numpy array.

    Returns
    -------
    np.ndarray or sparse.coo_matrix
        2D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.

    """
    if sparse_lhs:
        return sps.coo_matrix(
            (0, n_x) if A is None else A, dtype=np.float, copy=True
        )
    elif A is None:
        return np.zeros((0, n_x), dtype=np.float)
    else:
        return np.array(A, dtype=np.float, copy=True)

def _format_b_constraints(b):
    """Format the upper bounds of the constraints to a 1D array

    Parameters
    ----------
    b : 1D array
        1D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.

    Returns
    -------
    1D np.array
        1D array of values representing the upper-bound of each (in)equality
        constraint (row) in ``A``.

    """
    if b is None:
        return np.array([], dtype=np.float)
    b = np.array(b, dtype=np.float, copy=True).squeeze()
    return b if b.size != 1 else b.reshape((-1))

def _clean_inputs(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
                  x0=None):
    """
    Given user inputs for a linear programming problem, return the
    objective vector, upper bound constraints, equality constraints,
    and simple bounds in a preferred format.

    Parameters
    ----------
    c : 1D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2D array, optional
        2D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    b_ub : 1D array, optional
        1D array of values representing the upper-bound of each inequality
        constraint (row) in ``A_ub``.
    A_eq : 2D array, optional
        2D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    b_eq : 1D array, optional
        1D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    bounds : sequence, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative).
        If a sequence containing a single tuple is provided, then ``min`` and
        ``max`` will be applied to all variables in the problem.
    x0 : 1D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm.

    Returns
    -------
    c : 1D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2D array, optional
        2D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    b_ub : 1D array, optional
        1D array of values representing the upper-bound of each inequality
        constraint (row) in ``A_ub``.
    A_eq : 2D array, optional
        2D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    b_eq : 1D array, optional
        1D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative).
    x0 : 1D array, optional
        Starting values of the independent variables, which will be refined by
        the optimization algorithm.
    """
    if c is None:
        raise TypeError

    try:
        c = np.array(c, dtype=np.float, copy=True).squeeze()
    except ValueError:
        raise TypeError(
            "Invalid input for linprog: c must be a 1D array of numerical "
            "coefficients")
    else:
        # If c is a single value, convert it to a 1D array.
        if c.size == 1:
            c = c.reshape((-1))

        n_x = len(c)
        if n_x == 0 or len(c.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: c must be a 1D array and must "
                "not have more than one non-singleton dimension")
        if not(np.isfinite(c).all()):
            raise ValueError(
                "Invalid input for linprog: c must not contain values "
                "inf, nan, or None")

    sparse_lhs = sps.issparse(A_eq) or sps.issparse(A_ub)
    try:
        A_ub = _format_A_constraints(A_ub, n_x, sparse_lhs=sparse_lhs)
    except ValueError:
        raise TypeError(
            "Invalid input for linprog: A_ub must be a 2D array "
            "of numerical values")
    else:
        n_ub = A_ub.shape[0]
        if len(A_ub.shape) != 2 or A_ub.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_ub must have exactly two "
                "dimensions, and the number of columns in A_ub must be "
                "equal to the size of c")
        if (sps.issparse(A_ub) and not np.isfinite(A_ub.data).all()
                or not sps.issparse(A_ub) and not np.isfinite(A_ub).all()):
            raise ValueError(
                "Invalid input for linprog: A_ub must not contain values "
                "inf, nan, or None")

    try:
        b_ub = _format_b_constraints(b_ub)
    except ValueError:
        raise TypeError(
            "Invalid input for linprog: b_ub must be a 1D array of "
            "numerical values, each representing the upper bound of an "
            "inequality constraint (row) in A_ub")
    else:
        if b_ub.shape != (n_ub,):
            raise ValueError(
                "Invalid input for linprog: b_ub must be a 1D array; b_ub "
                "must not have more than one non-singleton dimension and "
                "the number of rows in A_ub must equal the number of values "
                "in b_ub")
        if not(np.isfinite(b_ub).all()):
            raise ValueError(
                "Invalid input for linprog: b_ub must not contain values "
                "inf, nan, or None")

    try:
        A_eq = _format_A_constraints(A_eq, n_x, sparse_lhs=sparse_lhs)
    except ValueError:
        raise TypeError(
            "Invalid input for linprog: A_eq must be a 2D array "
            "of numerical values")
    else:
        n_eq = A_eq.shape[0]
        if len(A_eq.shape) != 2 or A_eq.shape[1] != n_x:
            raise ValueError(
                "Invalid input for linprog: A_eq must have exactly two "
                "dimensions, and the number of columns in A_eq must be "
                "equal to the size of c")

        if (sps.issparse(A_eq) and not np.isfinite(A_eq.data).all()
                or not sps.issparse(A_eq) and not np.isfinite(A_eq).all()):
            raise ValueError(
                "Invalid input for linprog: A_eq must not contain values "
                "inf, nan, or None")
    try:
        b_eq = _format_b_constraints(b_eq)

    except ValueError:
        raise TypeError(
            "Invalid input for linprog: b_eq must be a 1D array of "
            "numerical values, each representing the upper bound of an "
            "inequality constraint (row) in A_eq")
    else:
        if b_eq.shape != (n_eq,):
            raise ValueError(
                "Invalid input for linprog: b_eq must be a 1D array; b_eq "
                "must not have more than one non-singleton dimension and "
                "the number of rows in A_eq must equal the number of values "
                "in b_eq")
        if not(np.isfinite(b_eq).all()):
            raise ValueError(
                "Invalid input for linprog: b_eq must not contain values "
                "inf, nan, or None")

    # x0 gives a (optional) starting solution to the solver. If x0 is None,
    # skip the checks. Initial solution will be generated automatically.
    if x0 is not None:
        try:
            x0 = np.array(x0, dtype=float, copy=True).squeeze()
        except ValueError:
            raise TypeError(
                "Invalid input for linprog: x0 must be a 1D array of "
                "numerical coefficients")
        if x0.ndim == 0:
            x0 = x0.reshape((-1))
        if len(x0) == 0 or x0.ndim != 1:
            raise ValueError(
                "Invalid input for linprog: x0 should be a 1D array; it "
                "must not have more than one non-singleton dimension")
        if not x0.size == c.size:
            raise ValueError(
                "Invalid input for linprog: x0 and c should contain the "
                "same number of elements")
        if not np.isfinite(x0).all():
            raise ValueError(
            "Invalid input for linprog: x0 must not contain values "
            "inf, nan, or None")

    # "If a sequence containing a single tuple is provided, then min and max
    # will be applied to all variables in the problem."
    # linprog doesn't treat this right: it didn't accept a list with one tuple
    # in it
    try:
        if isinstance(bounds, str):
            raise TypeError
        if bounds is None or len(bounds) == 0:
            bounds = [(0, None)] * n_x
        elif len(bounds) == 1:
            b = bounds[0]
            if len(b) != 2:
                raise ValueError(
                    "Invalid input for linprog: exactly one lower bound and "
                    "one upper bound must be specified for each element of x")
            bounds = [b] * n_x
        elif len(bounds) == n_x:
            try:
                len(bounds[0])
            except BaseException:
                bounds = [(bounds[0], bounds[1])] * n_x
            for i, b in enumerate(bounds):
                if len(b) != 2:
                    raise ValueError(
                        "Invalid input for linprog, bound " +
                        str(i) +
                        " " +
                        str(b) +
                        ": exactly one lower bound and one upper bound must "
                        "be specified for each element of x")
        elif (len(bounds) == 2 and np.isreal(bounds[0])
                and np.isreal(bounds[1])):
            bounds = [(bounds[0], bounds[1])] * n_x
        else:
            raise ValueError(
                "Invalid input for linprog: exactly one lower bound and one "
                "upper bound must be specified for each element of x")

        clean_bounds = []  # also creates a copy so user's object isn't changed
        for i, b in enumerate(bounds):
            if b[0] is not None and b[1] is not None and b[0] > b[1]:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": a lower bound must be less than or equal to the "
                    "corresponding upper bound")
            if b[0] == np.inf:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": infinity is not a valid lower bound")
            if b[1] == -np.inf:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": negative infinity is not a valid upper bound")
            lb = float(b[0]) if b[0] is not None and b[0] != -np.inf else None
            ub = float(b[1]) if b[1] is not None and b[1] != np.inf else None
            clean_bounds.append((lb, ub))
        bounds = clean_bounds
    except ValueError as e:
        if "could not convert string to float" in e.args[0]:
            raise TypeError
        else:
            raise e
    except TypeError as e:
        print(e)
        raise TypeError(
            "Invalid input for linprog: bounds must be a sequence of "
            "(min,max) pairs, each defining bounds on an element of x ")

    return c, A_ub, b_ub, A_eq, b_eq, bounds, x0



def _get_Abc(c,A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None,
             x0=None, undo=[],c0=0):
    """
    Given a linear programming problem of the form:
    Minimize::
        c @ x
    Subject to::
        A_ub @ x <= b_ub
        A_eq @ x == b_eq
         lb <= x <= ub
    where ``lb = 0`` and ``ub = None`` unless set in ``bounds``.
    Return the problem in standard form:
    Minimize::
        c @ x
    Subject to::
        A @ x == b
            x >= 0
    by adding slack variables and making variable substitutions as necessary.
    Parameters
    ----------
    c : 1D array
        Coefficients of the linear objective function to be minimized.
        Components corresponding with fixed variables have been eliminated.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    A_ub : 2D array, optional
        2D array such that ``A_ub @ x`` gives the values of the upper-bound
        inequality constraints at ``x``.
    b_ub : 1D array, optional
        1D array of values representing the upper-bound of each inequality
        constraint (row) in ``A_ub``.
    A_eq : 2D array, optional
        2D array such that ``A_eq @ x`` gives the values of the equality
        constraints at ``x``.
    b_eq : 1D array, optional
        1D array of values representing the RHS of each equality constraint
        (row) in ``A_eq``.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction. Bounds have been
        tightened where possible.
    x0 : 1D array
        Starting values of the independent variables, which will be refined by
        the optimization algorithm
    undo: list of tuples
        (`index`, `value`) pairs that record the original index and fixed value
        for each variable removed from the problem
    Returns
    -------
    A : 2D array
        2D array such that ``A`` @ ``x``, gives the values of the equality
        constraints at ``x``.
    b : 1D array
        1D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    x0 : 1D array
        Starting values of the independent variables, which will be refined by
        the optimization algorithm
    References
    ----------
    .. [9] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    """

    if sps.issparse(A_eq):
        sparse = True
        A_eq = sps.lil_matrix(A_eq)
        A_ub = sps.lil_matrix(A_ub)

        def hstack(blocks):
            return sps.hstack(blocks, format="lil")

        def vstack(blocks):
            return sps.vstack(blocks, format="lil")

        zeros = sps.lil_matrix
        eye = sps.eye
    else:
        sparse = False
        hstack = np.hstack
        vstack = np.vstack
        zeros = np.zeros
        eye = np.eye

    fixed_x = set()
    if len(undo) > 0:
        # these are indices of variables removed from the problem
        # however, their bounds are still part of the bounds list
        fixed_x = set(undo[0])
    # they are needed elsewhere, but not here
    bounds = [bounds[i] for i in range(len(bounds)) if i not in fixed_x]
    # in retrospect, the standard form of bounds should have been an n x 2
    # array. maybe change it someday.

    # modify problem such that all variables have only non-negativity bounds

    bounds = np.array(bounds)
    lbs = bounds[:, 0]
    ubs = bounds[:, 1]
    m_ub, n_ub = A_ub.shape

    lb_none = np.equal(lbs, None)
    ub_none = np.equal(ubs, None)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)

    # if preprocessing is on, lb == ub can't happen
    # if preprocessing is off, then it would be best to convert that
    # to an equality constraint, but it's tricky to make the other
    # required modifications from inside here.

    # unbounded below: substitute xi = -xi' (unbounded above)
    l_nolb_someub = np.logical_and(lb_none, ub_some)
    i_nolb = np.nonzero(l_nolb_someub)[0]
    lbs[l_nolb_someub], ubs[l_nolb_someub] = (
        -ubs[l_nolb_someub], lbs[l_nolb_someub])
    lb_none = np.equal(lbs, None)
    ub_none = np.equal(ubs, None)
    lb_some = np.logical_not(lb_none)
    ub_some = np.logical_not(ub_none)
    c[i_nolb] *= -1
    if x0 is not None:
        x0[i_nolb] *= -1
    if len(i_nolb) > 0:
        if A_ub.shape[0] > 0:  # sometimes needed for sparse arrays... weird
            A_ub[:, i_nolb] *= -1
        if A_eq.shape[0] > 0:
            A_eq[:, i_nolb] *= -1

    # upper bound: add inequality constraint
    i_newub = np.nonzero(ub_some)[0]
    ub_newub = ubs[ub_some]
    n_bounds = np.count_nonzero(ub_some)
    #A_ub = vstack((A_ub, zeros((n_bounds, A_ub.shape[1]))))
    #b_ub = np.concatenate((b_ub, np.zeros(n_bounds)))
    #A_ub[range(m_ub, A_ub.shape[0]), i_newub] = 1
    #b_ub[m_ub:] = ub_newub

    A_ub = vstack(( zeros((n_bounds, A_ub.shape[1])),A_ub))
    b_ub = np.concatenate((np.zeros(n_bounds),b_ub))
    A_ub[range(0, n_bounds), i_newub] = 1
    b_ub[:n_bounds] = ub_newub
    
    A1 = vstack((A_ub, A_eq))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((A_ub.shape[0],))))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros((A_ub.shape[0],))))
    # unbounded: substitute xi = xi+ + xi-
    l_free = np.logical_and(lb_none, ub_none)
    i_free = np.nonzero(l_free)[0]
    n_free = len(i_free)
    A1 = hstack((A1, zeros((A1.shape[0], n_free))))
    c = np.concatenate((c, np.zeros(n_free)))
    if x0 is not None:
        x0 = np.concatenate((x0, np.zeros(n_free)))
    A1[:, range(n_ub, A1.shape[1])] = -A1[:, i_free]
    c[np.arange(n_ub, A1.shape[1])] = -c[i_free]
    if x0 is not None:
        i_free_neg = x0[i_free] < 0
        x0[np.arange(n_ub, A1.shape[1])[i_free_neg]] = -x0[i_free[i_free_neg]]
        x0[i_free[i_free_neg]] = 0

    # add slack variables
    A2 = vstack([eye(A_ub.shape[0]), zeros((A_eq.shape[0], A_ub.shape[0]))])
    A = hstack([A1, A2])

    # lower bound: substitute xi = xi' + lb
    # now there is a constant term in objective
    i_shift = np.nonzero(lb_some)[0]
    lb_shift = lbs[lb_some].astype(float)
    c0 += np.sum(lb_shift * c[i_shift])
    if sparse:
        b = b.reshape(-1, 1)
        A = A.tocsc()
        b -= (A[:, i_shift] * sps.diags(lb_shift)).sum(axis=1)
        b = b.ravel()
    else:
        b -= (A[:, i_shift] * lb_shift).sum(axis=1)
    if x0 is not None:
        x0[i_shift] -= lb_shift

    return A, b, c, c0, x0    

def postsolve(x, n_x, complete=False, tol=1e-8, copy=False):
    """
    Given solution x to presolved, standard form linear program x, add
    fixed variables back into the problem and undo the variable substitutions
    to get solution to original linear program. Also, calculate the objective
    function value, slack in original upper bound constraints, and residuals
    in original equality constraints.

    Parameters
    ----------
    x : 1D array
        Solution vector to the standard-form problem.
    postsolve_args : tuple
        Data needed by _postsolve to convert the solution to the standard-form
        problem into the solution to the original problem, including:

        c : 1D array
            Original coefficients of the linear objective function to be
            minimized.
        A_ub : 2D array, optional
            2D array such that ``A_ub @ x`` gives the values of the upper-bound
            inequality constraints at ``x``.
        b_ub : 1D array, optional
            1D array of values representing the upper-bound of each inequality
            constraint (row) in ``A_ub``.
        A_eq : 2D array, optional
            2D array such that ``A_eq @ x`` gives the values of the equality
            constraints at ``x``.
        b_eq : 1D array, optional
            1D array of values representing the RHS of each equality constraint
            (row) in ``A_eq``.
        bounds : sequence of tuples
            Bounds, as modified in presolve
        undo: list of tuples
            (`index`, `value`) pairs that record the original index and fixed value
            for each variable removed from the problem

    complete : bool
        Whether the solution is was determined in presolve (``True`` if so)
    tol : float
        Termination tolerance; see [1]_ Section 4.5.

    Returns
    -------
    x : 1D array
        Solution vector to original linear programming problem
    fun: float
        optimal objective value for original problem
    slack : 1D array
        The (non-negative) slack in the upper bound constraints, that is,
        ``b_ub - A_ub @ x``
    con : 1D array
        The (nominally zero) residuals of the equality constraints, that is,
        ``b - A_eq @ x``
    lb : 1D array
        The lower bound constraints on the original variables
    ub: 1D array
        The upper bound constraints on the original variables
    """
    # note that all the inputs are the ORIGINAL, unmodified versions
    # no rows, columns have been removed
    # the only exception is bounds; it has been modified
    # we need these modified values to undo the variable substitutions
    # in retrospect, perhaps this could have been simplified if the "undo"
    # variable also contained information for undoing variable substitutions
    #c, A_ub, b_ub, A_eq, b_eq, bounds, undo = postsolve_args

    #n_x = len(c)

    # we don't have to undo variable substitutions for fixed variables that
    # were removed from the problem
    no_adjust = set()

    # if there were variables removed from the problem, add them back into the
    # solution vector
    '''
    if len(undo) > 0:
        no_adjust = set(undo[0])
        x = x.tolist()
        for i, val in zip(undo[0], undo[1]):
            x.insert(i, val)
        copy = True
    '''
    if copy:
        x = np.array(x, copy=True)

    # now undo variable substitutions
    # if "complete", problem was solved in presolve; don't do anything here
    if not complete and bounds is not None:  # bounds are never none, probably
        n_unbounded = 0
        for i, b in enumerate(bounds):
            if i in no_adjust:
                continue
            lb, ub = b
            if lb is None and ub is None:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                if lb is None:
                    x[i] = ub - x[i]
                else:
                    x[i] += lb
    x = x[:n_x]  # all the rest of the variables were artificial
    #comment out rest
    '''
    fun = x.dot(c)
    slack = b_ub - A_ub.dot(x)  # report slack for ORIGINAL UB constraints
    # report residuals of ORIGINAL EQ constraints
    con = b_eq - A_eq.dot(x)

    # Patch for bug #8664. Detecting this sort of issue earlier
    # (via abnormalities in the indicators) would be better.
    bounds = np.array(bounds)  # again, this should have been the standard form
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    lb[np.equal(lb, None)] = -np.inf
    ub[np.equal(ub, None)] = np.inf

    return x, fun, slack, con, lb, ub
    '''
    return x

def _initialization(shape, init_val=None):
    if init_val is None:
        m_eq, n = shape
        x0 = np.ones(n,dtype = np.float)
        y0 = np.zeros(m_eq,dtype = np.float)
        t0 = np.ones(n,dtype = np.float)
        tau0 = np.array([1], dtype=np.float)
        kappa0 = np.array([1], dtype=np.float)
    else:
        x0 = init_val['x']
        y0 = init_val['y']
        t0 = init_val['t']
        tau0 = init_val['tau']
        kappa0 = init_val['kappa']        
    return x0,y0,t0,tau0,kappa0

def _sym_solve(Dinv, A, r1, r2, solve):
    
    # [4] 8.31
    r = r2 + A.dot(Dinv * r1)

    #print(r)
    #print(solve)
    v =  solve(r)


    # try:
    # 	v = solve(r)
    # except:
    # 	print(r)
    # [4] 8.32
    u = Dinv * (A.T.dot(v) - r1)


    return u, v

def _get_solver(M, sparse=False, lstsq=False, sym_pos=True,
                cholesky=True, assume_a= 'sym',permc_spec='MMD_AT_PLUS_A'):
    """
    Given solver options, return a handle to the appropriate linear system
    solver.

    Parameters
    ----------
    M : 2D array
        As defined in [4] Equation 8.31
    sparse : bool (default = False)
        True if the system to be solved is sparse. This is typically set
        True when the original ``A_ub`` and ``A_eq`` arrays are sparse.
    lstsq : bool (default = False)
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : bool (default = True)
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system should be symmetric positive definite, due to
        numerical difficulties.
    cholesky : bool (default = True)
        True if the system is to be solved by Cholesky, rather than LU,
        decomposition. This is typically faster unless the problem is very
        small or prone to numerical difficulties.
    permc_spec : str (default = 'MMD_AT_PLUS_A')
        Sparsity preservation strategy used by SuperLU. Acceptable values are:

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering.

        See SuperLU documentation.

    Returns
    -------
    solve : function
        Handle to the appropriate solver function

    """
    try:
        if sparse:
            if lstsq:
                def solve(r, sym_pos=False):
                    return sps.linalg.lsqr(M, r)[0]
            elif cholesky:
                solve = (M)
            else:
                if has_umfpack and sym_pos:
                    solve = sps.linalg.factorized(M)
                else:  # factorized doesn't pass permc_spec
                    solve = sps.linalg.splu(M, permc_spec=permc_spec).solve
        else:
            if lstsq:  # sometimes necessary as solution is approached
                def solve(r):
                    return sp.linalg.lstsq(M, r)[0]
            elif cholesky:
                L = sp.linalg.cho_factor(M)

                def solve(r):
                    return sp.linalg.cho_solve(L, r)
            else:
                # this seems to cache the matrix factorization, so solving
                # with multiple right hand sides is much faster
                def solve(r, sym_pos=sym_pos,assume_a=assume_a):
                    return sp.linalg.solve(M, r,assume_a=assume_a)
    # There are many things that can go wrong here, and it's hard to say
    # what all of them are. It doesn't really matter: if the matrix can't be
    # factorized, return None. get_solver will be called again with different
    # inputs, and a new routine will try to factorize the matrix.
    except KeyboardInterrupt:
        raise
    except Exception:
    	return None
    return solve

def _get_delta(A,b,c,x,y,t,tau,kappa,gamma,eta,pc=True):
    n = len(x)
    # print("input TAU",tau)
    # print(" input X",x)
    r1 = -(A.dot(x) - b*tau) #r_p
    r2 = -(A.T.dot(y) + t - c*tau) #r_d
    r3 = -(-c.dot(x)+b.dot(y)-kappa) #r_g


    mu = (x.dot(t)+tau*kappa)/(n+1)
  
    Dinv = (x/t) 
    M = A.dot(Dinv.reshape(-1,1)*A.T)
    damping_param = 1e-6
    np.fill_diagonal(M, M.diagonal() + damping_param)

    solve = _get_solver(M)  
    lstsq = False
    # if solve is None:
    #     print(Dinv)
    #     print(A.shape)
    i =0
    while i <2:  

        rhat1 = eta(gamma)*r1
        rhat2 = eta(gamma)*r2
        rhat3 = eta(gamma)*r3
        rhatxt = -(x*t - gamma*mu) 
        rhattk = -(tau*kappa - gamma*mu)
        #alpha,
        #Mehrotra
        if i==1:
            rhatxt -=  d_x * d_t
            #rhatzs -= - d_z * d_s
            rhattk -=  d_tau*d_kappa
        attempt_count = 0
        solved = False
        while (not solved and attempt_count<3):
                try:
                    p,q = _sym_solve(Dinv,A,c,b,solve)
                    u,v = _sym_solve(Dinv,A,rhat2 -(1/x)*rhatxt,rhat1,solve)

                    
                    #print(M)
                    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    	raise (LinAlgError)
                    solved = True
                    logging.info("Success in finding delta ----- solving attempt %d "%(attempt_count))
                except (ValueError, TypeError,LinAlgError) as e:
                    logging.info("Cholesky failed trying other method ----- solving attempt %d error message %s"%(attempt_count,e))
                    
                    attempt_count +=1
                    # if not lstsq:
                    #     sym_pos = False
                    #     sparse= True
                    # else:
                    #     sym_pos = True
                    #     sparse =  True
                    assume_a_dict = {2:'pos',3:'gen',1:'sym'}
                    assume_a = assume_a_dict[attempt_count]
                    solve = _get_solver(M, cholesky = False,
                        assume_a =assume_a)

                    #print("Solving attempt: ",attempt_count)
                    
                    #if attempt_count ==3:
                        #print(e)
                        #print("Warning:: Not able to find delta matrix")
                        #logging.info("M is %s"%(M))

        if solved:
            ####################
            # print("Numerator ",
            #     ((rhat3 + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) ))
            # print("denomintor ",
            #     (1 / tau * kappa + (-c.dot(p) + b.dot(q))))                     
            d_tau = ((rhat3 + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) /
                     (1 / tau * kappa + (-c.dot(p) + b.dot(q))))
            d_x = u + p * d_tau
            d_y = v + q * d_tau
            d_t = (1/x)*(rhatxt - t*d_x)
            d_kappa = (rhattk - kappa * d_tau)/tau
            alpha = _get_step(x, d_x, t, d_t, tau, d_tau, kappa, d_kappa, 1)
            gamma =  (1- alpha)**2 *min(0.1,(1-alpha))
            i +=1
            if pc is not True:
                break
            # print ("grads")
            # print("d_x",d_x)
            # print("d_y",d_y)
            # print("d_t",d_t)
            # print("d_tau",d_tau)


        else:
            d_x,d_y,d_t,d_tau,d_kappa = 0. ,0.,0.,0.,0.
            break
        
    return d_x,d_y,d_t,d_tau,d_kappa,solved

def _do_step(x, y, t, tau, kappa,  d_x,d_y,d_t, d_tau, d_kappa, alpha):
    
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    t = t + alpha * d_t
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, t, tau, kappa

def _get_step(x, d_x, t, d_t, tau, d_tau, kappa, d_kappa, alpha0):
    i_x = d_x < 0
    i_t = d_t < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_t = alpha0 * np.min(t[i_t] / -d_t[i_t]) if np.any(i_t) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_t, alpha_kappa])
    return alpha

def _indicators(A, b, c,  x, y, t, tau, kappa):
   

    # residuals for termination are relative to initial values
    x0,y0,t0,tau0,kappa0 = _initialization(A.shape)

    # See [4], Section 4 - The Homogeneous Algorithm, Equation 8.8
    def r_p(x, tau):
        return b * tau - A.dot(x)

    def r_d(y, t, tau):
        return c * tau - A.T.dot(y) - t

    def r_g(x, y, kappa):
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    def mu(x, tau, t, kappa):
        return (x.dot(t) + np.dot(tau, kappa)) / (len(x) + 1)

    obj = c.dot(x / tau) 

    def norm(a):
        return np.linalg.norm(a)


    # See [4], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, t0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, t0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, t, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, t, kappa) / mu_0
    current_mu = mu(x, tau, t, kappa) 
    return rho_p, rho_d, rho_A, rho_g, rho_mu, current_mu,obj
def _presolve(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, rr, tol=1e-9):
    
    undo = []               # record of variables eliminated from problem
    # constant term in cost function may be added if variables are eliminated
    c0 = 0
    complete = False        # complete is True if detected infeasible/unbounded
    x = np.zeros(c.shape)   # this is solution vector if completed in presolve

    status = 0              # all OK unless determined otherwise
    message = ""

    # Standard form for bounds (from _clean_inputs) is list of tuples
    # but numpy array is more convenient here
    # In retrospect, numpy array should have been the standard
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    lb[np.equal(lb, None)] = -np.inf
    ub[np.equal(ub, None)] = np.inf
    bounds = bounds.astype(float)
    lb = lb.astype(float)
    ub = ub.astype(float)

    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape

    if (sps.issparse(A_eq)):
        A_eq = A_eq.tolil()
        A_ub = A_ub.tolil()

        def where(A):
            return A.nonzero()

        vstack = sps.vstack
    else:
        where = np.where
        vstack = np.vstack

    # zero row in equality constraints
    zero_row = np.array(np.sum(A_eq != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(
            np.logical_and(
                zero_row,
                np.abs(b_eq) > tol)):  # test_zero_row_1
            # infeasible if RHS is not zero
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding constraint value.")
            complete = True
            return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                    x, x0, undo, complete, status, message)
        else:  # test_zero_row_2
            # if RHS is zero, we can eliminate this equation entirely
            A_eq = A_eq[np.logical_not(zero_row), :]
            b_eq = b_eq[np.logical_not(zero_row)]

    # zero row in inequality constraints
    zero_row = np.array(np.sum(A_ub != 0, axis=1) == 0).flatten()
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
            # infeasible if RHS is less than zero (because LHS is zero)
            status = 2
            message = ("The problem is (trivially) infeasible due to a row "
                       "of zeros in the equality constraint matrix with a "
                       "nonzero corresponding  constraint value.")
            complete = True
            return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                    x, x0, undo, complete, status, message)
        else:  # test_zero_row_2
            # if LHS is >= 0, we can eliminate this constraint entirely
            A_ub = A_ub[np.logical_not(zero_row), :]
            b_ub = b_ub[np.logical_not(zero_row)]

    # zero column in (both) constraints
    # this indicates that a variable isn't constrained and can be removed
    A = vstack((A_eq, A_ub))
    if A.shape[0] > 0:
        zero_col = np.array(np.sum(A != 0, axis=0) == 0).flatten()
        # variable will be at upper or lower bound, depending on objective
        x[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):  # if an unconstrained variable has no bound
            status = 3
            message = ("If feasible, the problem is (trivially) unbounded "
                       "due  to a zero column in the constraint matrices. If "
                       "you wish to check whether the problem is infeasible, "
                       "turn presolve off.")
            complete = True
            return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                    x, x0, undo, complete, status, message)
        # variables will equal upper/lower bounds will be removed later
        lb[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]

    # row singleton in equality constraints
    # this fixes a variable and removes the constraint
    singleton_row = np.array(np.sum(A_eq != 0, axis=1) == 1).flatten()
    rows = where(singleton_row)[0]
    cols = where(A_eq[rows, :])[1]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                # infeasible if fixed value is not within bounds
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the equality constraints is "
                           "inconsistent with the bounds.")
                complete = True
                return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                        x, x0, undo, complete, status, message)
            else:
                # sets upper and lower bounds at that fixed value - variable
                # will be removed later
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]

    # row singleton in inequality constraints
    # this indicates a simple bound and the constraint can be removed
    # simple bounds may be adjusted here
    # After all of the simple bound information is combined here, get_Abc will
    # turn the simple bounds into constraints
    singleton_row = np.array(np.sum(A_ub != 0, axis=1) == 1).flatten()
    cols = where(A_ub[singleton_row, :])[1]
    rows = where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:  # upper bound
                if val < lb[col] - tol:  # infeasible
                    complete = True
                elif val < ub[col]:  # new upper bound
                    ub[col] = val
            else:  # lower bound
                if val > ub[col] + tol:  # infeasible
                    complete = True
                elif val > lb[col]:  # new lower bound
                    lb[col] = val
            if complete:
                status = 2
                message = ("The problem is (trivially) infeasible because a "
                           "singleton row in the upper bound constraints is "
                           "inconsistent with the bounds.")
                return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                        x, x0, undo, complete, status, message)
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]

    # identical bounds indicate that variable can be removed
    i_f = np.abs(lb - ub) < tol   # indices of "fixed" variables
    i_nf = np.logical_not(i_f)  # indices of "not fixed" variables

    # test_bounds_equal_but_infeasible
    if np.all(i_f):  # if bounds define solution, check for consistency
        residual = b_eq - A_eq.dot(lb)
        slack = b_ub - A_ub.dot(lb)
        if ((A_ub.size > 0 and np.any(slack < 0)) or
                (A_eq.size > 0 and not np.allclose(residual, 0))):
            status = 2
            message = ("The problem is (trivially) infeasible because the "
                       "bounds fix all variables to values inconsistent with "
                       "the constraints")
            complete = True
            return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
                    x, x0, undo, complete, status, message)

    ub_mod = ub
    lb_mod = lb
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x = x[i_nf]
        # user guess x0 stays separate from presolve solution x
        if x0 is not None:
            x0 = x0[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        # record of variables to be added back in
        undo = [np.nonzero(i_f)[0], lb[i_f]]
        # don't remove these entries from bounds; they'll be used later.
        # but we _also_ need a version of the bounds with these removed
        lb_mod = lb[i_nf]
        ub_mod = ub[i_nf]

    # no constraints indicates that problem is trivial
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        # test_empty_constraint_1
        if c.size == 0:
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        elif (np.any(np.logical_and(c < 0, ub_mod == np.inf)) or
              np.any(np.logical_and(c > 0, lb_mod == -np.inf))):
            # test_no_constraints()
            # test_unbounded_no_nontrivial_constraints_1
            # test_unbounded_no_nontrivial_constraints_2
            status = 3
            message = ("The problem is (trivially) unbounded "
                       "because there are no non-trivial constraints and "
                       "a) at least one decision variable is unbounded "
                       "above and its corresponding cost is negative, or "
                       "b) at least one decision variable is unbounded below "
                       "and its corresponding cost is positive. ")
        else:  # test_empty_constraint_2
            status = 0
            message = ("The solution was determined in presolve as there are "
                       "no non-trivial constraints.")
        complete = True
        x[c < 0] = ub_mod[c < 0]
        x[c > 0] = lb_mod[c > 0]
        # where c is zero, set x to a finite bound or zero
        x_zero_c = ub_mod[c == 0]
        x_zero_c[np.isinf(x_zero_c)] = ub_mod[c == 0][np.isinf(x_zero_c)]
        x_zero_c[np.isinf(x_zero_c)] = 0
        x[c == 0] = x_zero_c
        # if this is not the last step of presolve, should convert bounds back
        # to array and return here

    # *sigh* - convert bounds back to their standard form (list of tuples)
    # again, in retrospect, numpy array would be standard form
    lb[np.equal(lb, -np.inf)] = None
    ub[np.equal(ub, np.inf)] = None
    bounds = np.hstack((lb[:, np.newaxis], ub[:, np.newaxis]))
    bounds = bounds.tolist()
    for i, row in enumerate(bounds):
        for j, col in enumerate(row):
            if str(col) == "nan":
                # comparing col to float("nan") and np.nan doesn't work.
                # should use np.isnan
                bounds[i][j] = None
    return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
            x, x0, undo, complete, status, message)
def _remove_redundant_rows (A_eq):
    # remove redundant (linearly dependent) rows from equality constraints
    n_rows_A = A_eq.shape[0]
    redundancy_warning = ("A_eq does not appear to be of full row rank. To "
                          "improve performance, check the problem formulation "
                          "for redundant equality constraints.")
    # if (sps.issparse(A_eq)):
    #     if rr and A_eq.size > 0:  # TODO: Fast sparse rank check?
    #         A_eq, b_eq, status, message = _remove_redundancy_sparse(A_eq, b_eq)
    #         if A_eq.shape[0] < n_rows_A:
    #             warn(redundancy_warning, OptimizeWarning, stacklevel=1)
    #         if status != 0:
    #             complete = True
    #     return (c, c0, A_ub, b_ub, A_eq, b_eq, bounds,
    #             x, x0, undo, complete, status, message)

    # This is a wild guess for which redundancy removal algorithm will be
    # faster. More testing would be good.
    small_nullspace = 5
    if  A_eq.size > 0:
        try:  # TODO: instead use results of first SVD in _remove_redundancy
            rank = np.linalg.matrix_rank(A_eq)
        except Exception:  # oh well, we'll have to go with _remove_redundancy_dense
            rank = 0
    if A_eq.size > 0 and rank < A_eq.shape[0]:
        warn(redundancy_warning, OptimizeWarning, stacklevel=3)
        dim_row_nullspace = A_eq.shape[0]-rank
        if dim_row_nullspace <= small_nullspace:
            print("1")
            d_removed,  status, message = _remove_redundancy(A_eq)
        if dim_row_nullspace > small_nullspace :
            print("2")
            d_removed,  status, message = _remove_redundancy_dense(A_eq)
        if A_eq.shape[0] < rank:
            message = ("Due to numerical issues, redundant equality "
                       "constraints could not be removed automatically. "
                       "Try providing your constraint matrices as sparse "
                       "matrices to activate sparse presolve, try turning "
                       "off redundancy removal, or try turning off presolve "
                       "altogether.")
            status = 4
        if status != 0:
            complete = True
    return d_removed

def _preprocess(c, A=None, b=None, G=None, h=None, bounds=None):
    
    assert (A is not None or G is not None) and (b is not None or h is not None)

    if A is not None:
        n_eq = A.shape[0]
    else:
        n_eq = 0
    if G is not None:
        n_ub = G.shape[0]
    else:
        n_ub =0
    
    c, A_ub, b_ub, A_eq, b_eq, bounds, x0 = _clean_inputs(c=c, A_ub=G, b_ub =h,A_eq=A,b_eq=b,
     bounds = bounds)
    n_x = len(c)
    # ### presolve
    # c0 = 0
    # iteration = 0
    # complete = False    # will become True if solved in presolve
    # undo = []
    # n_x = len(c)
    # ### presolve
    # if presolve:
    #     (c, c0, A_ub, b_ub, A_eq, b_eq, bounds, x, x0, undo, complete, status,
    #         message) = _presolve(c, A_ub, b_ub, A_eq, b_eq, bounds, x0, rr= A_eq_rr, tol=1e-9)
    # if not complete:
    #     A, b, c, c0, x0 = _get_Abc(c, c0, A_ub, b_ub, A_eq,
    #                                b_eq, bounds, x0, undo)
    #     if c0 != 0:
    #         print("c0 is not zero ",c0)
    return c, A_ub, b_ub, A_eq, b_eq, bounds, x0,n_x,n_eq,n_ub


def _postprocess(x, postsolve_args,complete):
    ### BIG BUGS
    #print("n_x %d n_ub %d n_eq %d "%(n_x,n_ub,n_eq))
    c, A_ub, b_ub, A_eq, b_eq, bounds, undo = postsolve_args 
    n_x = len(c)
    no_adjust = set()


    # if there were variables removed from the problem, add them back into the
    # solution vector
    if len(undo) > 0:
        no_adjust = set(undo[0])
        x = x.tolist()
        for i, val in zip(undo[0], undo[1]):
            x.insert(i, val)
        
        x = np.array(x, copy=True)

    # now undo variable substitutions
    # if "complete", problem was solved in presolve; don't do anything here
    if not complete and bounds is not None:  # bounds are never none, probably
        n_unbounded = 0
        for i, bnds in enumerate(bounds):
            if i in no_adjust:
                continue
            lb, ub = bnds
            if lb is None and ub is None:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                if lb is None:
                    x[i] = ub - x[i]
                else:
                    x[i] += lb
    if not complete:
        # A =  A[-(n_ub+n_eq):,:n_x]
        # # A_ub = A[:n_ub,:]
        # # A_eq = A[n_ub:,:]
        # b = b[-(n_ub+n_eq):]
        # # b_ub = b[:n_ub]
        # # b_eq = b[n_ub:]        
        # c = c[:n_x]
        x = x[:n_x]
        
    return x   # x,y,c_v,A_v, b_v, x_v, t_v

def IPOfunc(A =None,b =None,h=None,cTrue=None,GTrue=None,purchase_fee=0,compensation_fee=0,alpha0=0.9995,beta=0.1,pc = True,
    tol= 1e-8,max_iter=1000,init_val= None,method =1,mu0=None,
    smoothing=False,bounds= None,thr=None,new_grad=True,presolve= True, A_eq_rr = True,
    damping= 1e-2):
    run_time = 0.
    problem_solved = True
    # if no solution under timelimit or max-iter don't do gradient update
    class IPOfunc_cls(Function):        
        @staticmethod
        def forward(ctx,cGTemp):
#            print(cGTemp.shape)  # torch.Size([10, 2])
            itemNum = len(h) - 1
            G = torch.zeros((itemNum+1, itemNum))
            c = - purchase_fee * cGTemp[:, 0]
            for i in range(itemNum):
                G[i][i] = 1
            G[itemNum] = cGTemp[:, 1]
#            time_start=time.time()
            #print("forward")
            nonlocal run_time
            start = time.time()
            c_=c.detach().numpy()
            if b is not None:
                b_=b.detach().numpy()
                if b_.shape[0] ==0:
                    b_ = None
            if A is not None:
                A_ = A.detach().numpy()
                if A_.shape[0] ==0:
                    A_ = None

            if h is not None:
                h_ = h.detach().numpy()
                if h_.shape[0] ==0:
                    h_ = None
            if G is not None:
                G_ = G.detach().numpy()
                if G_.shape[0] ==0:
                    G_ = None
            n = len(c_)
            bounds_ = bounds if bounds is not None else [(0.,None) for i in range(n)]
            thr_ = thr if thr is not None else 0.
            max_iter_ = max_iter if max_iter is not None else 1000
            # presolve_= presolve
            # A_eq_rr_ = A_eq_rr
            try:
                c_, G_, h_,A_, b_,bounds_, x0,n_x,n_eq,n_ub = _preprocess(c_, A_, b_, G_, 
                    h_,bounds_ )
                # print("After pre process shape of c {} of A{} b{} h{} G{}".format(c_.shape,
                #     A_.shape,b_.shape,h_.shape,G_.shape))
                c_o, A_o, b_o, G_o, h_o = c_.copy(), A_.copy(
                    ), b_.copy(), G_.copy(), h_.copy()
                # logging.info("Shape of A_eq {} A_ineq {} at the beginning".format(A_.shape,G_.shape))

            except ValueError:
                
                # c contains  inf, nan, or None
                print(c_, G_, h_,A_, b_)
                logging.info("####### Alert #########")
                logging.info("matrix contains inf, nan, or None")
                logging.info("C containes nan ? {} Infinity? {} ".format(np.isnan(c_).any(),
                np.isinf(c_).any()))
                logging.info("smoothing {} thr {}".format(smoothing,thr))
                logging.info("####### #########")
                raise (LinAlgError)
                # c_ = np.nan_to_num(c_)
                # c_, G_, h_,A_, b_,bounds_, x0,n_x,n_eq,n_ub = _preprocess(c_, A_, b_, G_, h_,bounds_)
            ### presolve
            
            c0 = 0
            iteration = 0
            complete = False    # will become True if solved in presolve
            undo = []
            ### presolve
            # print("Preprocess done{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))

            if presolve:
                # rows_to_be_removed = _remove_redundant_rows(A_)
                # A_ = np.delete(A_, rows_to_be_removed, axis=0)
                # b_ = np.delete(b_, rows_to_be_removed)
                (c_, c0, G_, h_, A_, b_, bounds_, x, x0, undo, complete, status,
                    message) = _presolve(c_, G_, h_,A_, b_,bounds_, x0, rr= A_eq_rr, tol=1e-9)
                # logging.info("After pre solve shape of c {} of A{} b{} h{} G{}".format(c_.shape,
                #     A_.shape,b_.shape,h_.shape,G_.shape))
            if len(undo)>0:
                ctx.fixed_var = undo[0]
            else:
                ctx.fixed_var = []
            # print("Presolve done{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
            n_eq_afterpre = len(b_)
            n_ub_afterpre = len(h_)
            n_x_afterpre = len(c_)

            all_index = set([i for i in range(n_x)])
            ctx.var_index= list(all_index.difference(ctx.fixed_var))
            ctx.n_x = n_x
            ctx.n_eq = n_eq
            ctx.n_ub = n_ub
            
            postsolve_args = (c_o,  A_o, b_o, G_o, h_o, bounds, undo)

            if not complete:
                # logging.info("Shape of A_eq {} A_ineq {} before getabc".format(A_.shape,G_.shape))
                A_, b_, c_, c0, x0 = _get_Abc(c_, G_, h_,A_, b_,bounds_, x0, undo,c0)
                # logging.info("Shape of A after getabc {}".format(A_.shape))
                if c0 != 0:
                    logging.info("c0 is not zero! {} ".format(c0))
               
                x,y,t,tau,kappa = _initialization(A_.shape,init_val)
                go = True
                iter_count = 0 
                # print("Go loop to start{:%Y-%m-%d %H:%M:%S}".format(datetime.datetime.now()))
                while go:
                    iter_count+=1
                    # print("iteration -",iter_count)
                    gamma = 0 if pc else beta * np.mean(t * x)
                    def eta(g=gamma):
                        return 1 - g
                    d_x,d_y,d_t, d_tau, d_kappa,solved = _get_delta(A_,b_,c_,x,y,t,tau,kappa,gamma,eta,pc= pc)
                
                    #logging.info("d_x %s  ,d_tau  %s, d_kappa %s"%(d_x,d_tau, d_kappa))
                    #print(solved)
                    #if not solved:
                    #    logging.info(" G %s, h %s, c %s" %(G,h,c))

                    alpha = _get_step(x, d_x, t, d_t, tau, d_tau, kappa, d_kappa, alpha0)
                    x,y,t,tau,kappa = _do_step(x, y, t, tau, kappa,  d_x,d_y,d_t, d_tau, d_kappa, alpha)

                    rho_p, rho_d, rho_A, rho_g, rho_mu,mu, obj = _indicators(
                        A_, b_, c_,  x, y, t, tau, kappa)

                    go = (rho_p > tol or rho_d > tol or rho_A > tol) and  (mu> thr_)
                    inf1 = (rho_p < tol and rho_d < tol and rho_g < tol and tau < tol *max(1, kappa))
                    inf2 = rho_mu < tol and tau < tol * min(1, kappa)
                    #print("alpha",alpha,"rhop",rho_p,"rhod",rho_d,"rhoA",rho_A,"rhog",rho_g,"rho_mu",rho_mu,"tau",tau)
                    if inf1 or inf2:
                        break
                    if (max_iter_ < iter_count):
                        logging.info("maximum iteration reached")
                        problem_solved = False
                        break

                    if not solved:
                        #print("no solution available")
                        logging.info("not able to find solutions")
                        problem_solved = False
                        break
                    #print(np.sum(x_hat))
                    #print(c.dot(x_hat))
                    ### experimnets with dividing by tau
                    #x = x/tau
                    #y = y/tau
                    #t = t/tau
                    #tau = 1
                save_for_initialization ={"x":x,"y":y,"t":t,"tau":tau,"kappa":kappa}
                logging.info("stopping mu value %s and threshold value is %s stopping tau %s and stopping kappa %s after iter count %d" %(mu,
                    thr_,tau,kappa, iter_count))
                if kappa>1e-3:
                    logging.info("kappa >0 ; c.x {}, b.y {}".format(c_.dot(x),
                        b_.dot(y)))

                # logging.info("calculated mu {:.2f}".format((x.dot(t) + np.dot(tau, kappa)) / (len(x) + 1)))

                
                x_hat = x/tau
                t_hat = t/tau
                y_hat = y/tau
                # logging.info("calculated mu {:.2f}".format((x_hat.dot(t_hat) ) / (len(x) + 1)))
                x_v = torch.from_numpy(x).float()
                y_v = torch.from_numpy(y).float()
                t_v = torch.from_numpy(t).float()                
                #print("y full",y_hat)
                # two choices
                # solve inequality problem or solve the problem with equality in backward 

                # x_sol,y_v, x_v,t_v = _postprocess(x_hat,y_hat,t_hat,
                #     n_x,n_eq,n_ub,bounds_ ,undo,complete)
                x_sol  = _postprocess(x_hat, postsolve_args,complete)
#                x_sol = torch.from_numpy(x_sol).float()
                

                c_v = torch.from_numpy(c_).float()
                A_v = torch.from_numpy(A_).float()
                b_v = torch.from_numpy(b_).float()
                
                global x_s1
                x_s1 = torch.from_numpy(x_sol)
#                x_s1 = x_sol

#                rowSizeA = A.shape[0]
                rowSizeG = G.shape[0]
                c_positive = -cTrue
                c_list = c_positive.detach().tolist()
#                A_list = A.detach().tolist()
#                b_list = b.detach().tolist()
                h_list = h.detach().tolist()
                G_list = GTrue.tolist()
#                pred_c_positive = -cGTemp[:, 0]
#                pred_c_list = pred_c_positive.detach().tolist()
#                pred_G = cGTemp[:, 1]
#                pred_G_list = pred_G.detach().tolist()
#
#                # Stage 1:
#                m1 = gp.Model()
#                m1.setParam('OutputFlag', 0)
#                x_m1 = m1.addVars(varNum, vtype=GRB.BINARY, name='x')
#                m1.setObjective(purchase_fee * x_m1.prod(pred_c_list), GRB.MAXIMIZE)
#                m1.addConstr((x_m1.prod(pred_G_list)) <= cap)
#                m1.optimize()
#                global x_s1
#                x_s1 = np.zeros(varNum,dtype='i')
#                predSol = np.zeros(varNum,dtype='i')
#                for i in range(varNum):
#                    x_s1[i] = x_m1[i].x
#                    predSol[i] = x_m1[i].x
#                x_s1 = torch.from_numpy(x_s1)
#
                # Stage 2:
                m2 = gp.Model()
                m2.setParam('OutputFlag', 0)
#                x_m2 = m2.addVars(varNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='x')
                y_m2 = m2.addVars(varNum, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='y')
#                x_m2 = m2.addVars(varNum, vtype=GRB.BINARY, name='x')
#                y_m2 = m2.addVars(varNum, vtype=GRB.BINARY, name='y')

                OBJ = 0
                for i in range(varNum):
                    OBJ = OBJ + purchase_fee * c_list[i] * x_sol[i] - (compensation_fee - purchase_fee) * c_list[i] * y_m2[i]
                m2.setObjective(OBJ, GRB.MAXIMIZE)
                
#                for i in range(rowSizeA):
#                    m2.addConstr((x_m2.prod(A_list[i])) == b_list[i])
#                for i in range(rowSizeG):
                has_selected_weight = 0
                for i in range(varNum):
                    has_selected_weight = has_selected_weight + x_sol[i] * G_list[rowSizeG-1][i]
                m2.addConstr(has_selected_weight - (y_m2.prod(G_list[rowSizeG-1])) <= h_list[rowSizeG-1])
#                m2.addConstrs(x_m2[i] == x_sol[i] for i in range(varNum))
                m2.addConstrs(y_m2[i] <= x_sol[i] for i in range(varNum))
                
#                print(c_list, G_list, h_numpy, x_sol)
                m2.optimize()
                m2Objective = m2.objVal
                x_prime = np.zeros(varNum)
                t_sol = np.zeros(varNum)
                for i in range(varNum):
                    x_prime[i] = x_sol[i] - y_m2[i].x
                    t_sol[i] = y_m2[i].x
                s_sol = np.zeros(2*varNum)
                for i in range(varNum):
                    s_sol[i] = t_sol[i] - (x_prime[i] - x_sol[i])
                for j in range(varNum, 2*varNum):
                    s_sol[j] = t_sol[j-varNum] - (x_sol[i-varNum] - x_prime[i-varNum])
#                except:
#                    print(c_list, G_list, x_sol)
                
#                print(x_prime, s_sol, t_sol)
                x_sol = torch.from_numpy(x_prime)
                end = time.time()
                run_time += end -start
#                print("violateFactor: ", violateFactor, "violateEdgeIndex: ", violateEdgeIndex)
                
                if method==1: # full

                    ctx.save_for_backward(x_v,y_v,t_v,
                        torch.tensor(tau,dtype=torch.float),
                        torch.tensor(kappa,dtype=torch.float),
                        c_v, A_v, b_v, A, b, G, h,
                        torch.from_numpy(x_prime),
                        torch.from_numpy(s_sol),
                        torch.from_numpy(t_sol))
                if method==2: #both row & columns
                    ctx.save_for_backward(x_v[:n_x_afterpre],
                        y_v[-(n_ub_afterpre + n_eq_afterpre):],t_v[:n_x_afterpre],
                        torch.tensor(tau,dtype=torch.float),
                        torch.tensor(kappa,dtype=torch.float),
                        c_v[:n_x_afterpre],
                        A_v[-(n_ub_afterpre + n_eq_afterpre):,:n_x_afterpre],
                        b_v[-(n_ub_afterpre + n_eq_afterpre):],
                        A,
                        b,
                        G,
                        h,
                        torch.from_numpy(x_prime),
                        torch.from_numpy(s_sol),
                        torch.from_numpy(t_sol))
                if method==3: #only rows
                    ctx.save_for_backward(x_v,
                        y_v[-(n_ub_afterpre + n_eq_afterpre):],t_v,
                        torch.tensor(tau,dtype=torch.float),
                        torch.tensor(kappa,dtype=torch.float),
                        c_v,
                        A_v[-(n_ub_afterpre + n_eq_afterpre):,:],
                        b_v[-(n_ub_afterpre + n_eq_afterpre):],
                        A,
                        b,
                        G,
                        h,
                        torch.from_numpy(x_prime),
                        torch.from_numpy(s_sol),
                        torch.from_numpy(t_sol))
            else:
                x_sol = torch.from_numpy(x).float()
                t_sol = torch.ones(n_x)
                y_sol = torch.zeros(n_eq)
                z_sol = torch.zeros(n_ub)
                print("alert solved in presolve not sure about dual vaialbles")

        
            # c = torch.from_numpy(c).float()
            # A = torch.from_numpy(A).float()
            # b = torch.from_numpy(b).float()
            # G = torch.from_numpy(G).float()
            # h = torch.from_numpy(h).float()
#            print(x_sol,violateFactor,violateEdgeIndex)
#            print(x_sol)
#            print("LP solved")
#            time_end1=time.time()
#            print('Solve LP cost:',time_end1-time_start,'s')
            return x_sol
            
        @staticmethod
        def backward(ctx,del_x):
            time_end1=time.time()
            #print(del_x)
            #print("backward")
            nonlocal run_time
            start = time.time()
            x,y,t,tau,kappa,c,A,b,trueA,trueb,trueG,trueh,x_sol,s_sol,t_sol = ctx.saved_tensors
#            print(x_sol,violateFactor,violateEdgeIndex)
#            print("violateFactor: ", violateFactor)
            logging.info("shape of x {} y {} t {} c {} A {} b {}".format(x.shape,
                y.shape,t.shape,c.shape,A.shape,b.shape))
            #print("shape of x {} y {} t {} tau {} kappa {} c {} A {} b {}".format(x.shape, y.shape,t.shape, tau.shape, kappa.shape, c.shape,A.shape,b.shape))
            #shape of x torch.Size([285]) y torch.Size([43]) t torch.Size([285]) tau torch.Size([1]) kappa torch.Size([1]) c torch.Size([285]) A torch.Size([43, 285]) b torch.Size([43])
            #shape of x torch.Size([50]) y torch.Size([2]) t torch.Size([50]) tau torch.Size([1]) kappa torch.Size([1]) c torch.Size([50]) A torch.Size([2, 50]) b torch.Size([2])

            #print(x)
            #print(A)
            #print(b)
            #print(t)
            #print(tau)
            #print(kappa)
            n  = len(x)
            mu = (x.dot(t) + tau*kappa) / (n + 1)
            del_x = del_x[ctx.var_index]
            assert len(del_x)== (ctx.n_x - len(ctx.fixed_var))
            n_var = len(del_x)
            #assert n_var == n
            
            GSize = n - n_var
            G = A[:GSize,]
            h = b[:GSize]
            #print("shape of x {} y {} t {} tau {} kappa {} c {} A {} b {} G {} h {}".format(x.shape, y.shape,t.shape, tau.shape, kappa.shape, c.shape,A.shape,b.shape,G.shape,h.shape))
            #shape of x torch.Size([50]) y torch.Size([2]) t torch.Size([50]) tau torch.Size([1]) kappa torch.Size([1]) c torch.Size([50]) A torch.Size([2, 50]) b torch.Size([2])
            #shape of x torch.Size([86]) y torch.Size([66]) t torch.Size([86]) tau torch.Size([1]) kappa torch.Size([1]) c torch.Size([86]) A torch.Size([66, 86]) b torch.Size([66]) G torch.Size([43, 86]) h torch.Size([43])

            #print(x)
            #print(c)
            #print(A)
            #print(b)
            #print(G)
            #print(h)
            #print("t: ", t)
            #print("tau: ", tau)
            #print("kappa: ", kappa)
            laplace_smoothing = damping
            # db =  torch.zeros_like(b,dtype = torch.float)
            # dh = torch.zeros_like(h,dtype = torch.float)
            # dA =  torch.zeros_like(A,dtype = torch.float)
            # dG = torch.zeros_like(G,dtype = torch.float)
            
            logging.info("Value of mu: {:f} and n {}".format(mu.item(),n))
            
            for i in range(varNum):
                x[i] = x_sol[i]
            for i in range(varNum):
                A[varNum][i] = GTrue[varNum][i]
#            print(A)
            x2= x**2
            a = A*x
            M = torch.matmul(a,a.T)

            r1 = tau*A*x2
            r2 =  mu *b + torch.mv(A,x2*c)
            try:
                M_numpy = M.detach().numpy()
                # rank = np.linalg.matrix_rank(M_numpy,tol=1e-5)

                np.fill_diagonal(M_numpy, M_numpy.diagonal() + laplace_smoothing)
                rank = np.linalg.matrix_rank(M_numpy,tol=1e-3)
                logging.info("rank of M  {}and shape of M {}".format(rank, M_numpy.shape))
                r1_numpy = r1.detach().numpy()
                r2_numpy = r2.detach().numpy()
                ## will it be M????###
                v_numpy = sp.linalg.solve( M_numpy ,r1_numpy ,assume_a='sym')
                q_numpy = sp.linalg.solve( M_numpy ,r2_numpy ,assume_a='sym')
                v = torch.from_numpy(v_numpy).float()
                q = torch.from_numpy(q_numpy).float()

            except:
                raise (LinAlgError)
            p = x2* ( torch.mv(torch.t(A),q) - c)/mu
            u = x2.view(-1,1)* ( torch.matmul(torch.t(A),v) - tau*torch.eye(n))/mu
            # logging.info("shape:: p  {} q  {} u {} v {}".format(p.shape, q.shape,
            #     u.shape, v.shape))
            #print("shape:: p  {} q  {} u {} v {}".format(p.shape, q.shape, u.shape, v.shape))
            # shape:: p  torch.Size([50]) q  torch.Size([2]) u torch.Size([50, 50]) v torch.Size([2, 50])
            dtau = x + torch.mv(torch.t(u),c) -torch.mv(torch.t(v),b)
            dtau = dtau/(b.dot(q) - c.dot(p)+ (mu/tau**2))
            #print("shape:: dtau  {}".format(dtau.shape))  # shape:: dtau  torch.Size([50])
            dxc = u + torch.einsum('i,j->ij', p, dtau)
            #print("shape:: dxc  {}".format(dxc.shape))  # shape:: dxc  torch.Size([50, 50])
            dxc_cut = dxc[0:n_var,0:n_var]
            dxc_numpy = dxc_cut.numpy()
            #print(n_var) # 48
            #print("shape:: dxc_cut  {} del_x  {}".format(dxc_cut.shape, del_x.shape))
            # shape:: dxc_cut  torch.Size([48, 48]) del_x  torch.Size([48])
            #print(x)
            #print(del_x)
#            dc = torch.mv(dxc_cut, del_x)
            #print("shape:: dc  {}".format(dc.shape))  # shape:: dc  torch.Size([48])
            #print("dc ",dc)

#            c_grad = torch.zeros(ctx.n_x,dtype=torch.float)
#            c_grad[ctx.var_index] = dc
#            print(c_grad.shape, dc.shape)  # torch.Size([43]) torch.Size([43])

#            if any(torch.isnan(c_grad).tolist()):
#                logging.info("nan in bkwd pass ; del_x contains NaN?- {}".format(any(torch.isnan(del_x).tolist())))
#            if any(torch.isinf(c_grad).tolist()):
#                logging.info("Inf in bkwd pass ; del_x contains Inf?- {}".format(any(torch.isinf(del_x).tolist())))
            # db =  torch.zeros_like(b,dtype = torch.float)
            # dh = torch.zeros_like(h,dtype = torch.float)
            # dA =  torch.zeros_like(A,dtype = torch.float)
            # dG = torch.zeros_like(G,dtype = torch.float)
        
            n  = len(x) # n+p
            mu = (x.dot(t) + tau*kappa) / (n + 1)
            del_x = del_x[ctx.var_index]
            assert len(del_x)== (ctx.n_x - len(ctx.fixed_var))
            n_var = len(del_x)  # n
#            print(n, n_var) #97 48
            
#            colSizeG = n_var   # n
#            rowSizeG = n - n_var    # p
            rowSizeG = np.size(trueG, 0)
            colSizeG = np.size(trueG, 1)
#            print(rowSizeG, colSizeG)
            colSizeA = n_var   # n
            rowSizeA = len(y) - rowSizeG    # len(y)=m+p
            trueX = x[:n_var]   # n
            trueX = x_sol
#            print(trueA)
            if torch.sum(trueA) == 0:
                trueA = torch.zeros(0, n_var)
#            print(x_sol.shape)
#            trueG = A[:rowSizeG,:n_var]    # p*n
#            trueh = b[:rowSizeG]           # p
#            print(trueh)
#            trueA = A[rowSizeG:,:n_var]    # m*n
#            print(trueG.shape) # torch.Size([2, 48]) torch.Size([43, 242])
#            print(trueh.shape) # torch.Size([2]) torch.Size([43])
#            print(trueA.shape) # torch.Size([0, 48]) torch.Size([0, 242])
            #assert n_var == n
            
            logging.info("Value of mu: {:f} and n {}".format(mu.item(),n))
            x_numpy = trueX.numpy()
            G_numpyTemp = trueG.numpy()
            h_numpyTemp = trueh.numpy()
            A_numpy = trueA.numpy()
#            print(G_numpyTemp.shape, h_numpyTemp.shape) # (11, 10) (11,)
            
            G_numpy = np.zeros((1, rowSizeG-1))
            for i in range(rowSizeG-1):
                G_numpy[0][i] = G_numpyTemp[rowSizeG-1][i]
            h_numpy = h_numpyTemp[rowSizeG-1]
            rowSizeG = 1
#            print(n_var)
            H = np.zeros((n_var, n_var))
            for i in range(n_var):
                H[i][i] = mu / (x_numpy[i]*x_numpy[i])
            #print(H.shape)  # (48, 48) (242, 242)
            hGx = h_numpy - G_numpy.dot(x_numpy)
            #print(hGx.shape)    # (2,) (43,) (43,)
            hGx2 = 1 / (hGx*hGx)
#            hGx2 = np.zeros((rowSizeG, rowSizeG))
#            for i in range(rowSizeG):
#                hGx2[i][i] = 1 / (hGx*hGx)
#            print(G_numpy.T.shape)
#            H2 = G_numpy.T.dot(hGx2).dot(G_numpy)
            H2 = hGx2*G_numpy.T.dot(G_numpy)
            H2 = mu * H2
#            print(H2.shape)    #torch.Size([48, 48]) torch.Size([43, 43]) torch.Size([242, 242])
            for i in range(n_var):
                H[i][i] = H[i][i] + H2[i][i]
            H = np.linalg.inv(H)
#            AHA = A_numpy.dot(H).dot(A_numpy.T)
#            AHA_inverse = np.linalg.inv(AHA)
#            dxGTemp = H.dot(A_numpy.T).dot(AHA_inverse).dot(A_numpy).dot(H) - H
            dxGTemp = - H
            
#            time_end2=time.time()
#            print('Compute dxGTemp cost:',time_end2-time_end1,'s')
            
            #dxc = -H
#            print(dxh.shape)    #(43, 43) (242, 242)
#            fGxE = mu/hGx * np.eye(n_var)
#            print(fGxE)
#            print(fGxE.shape)
            fGx = mu*hGx2*G_numpy.dot(x_numpy) + mu/hGx * np.eye(n_var)
#            print(G_numpy.shape,h_numpy.shape,fGx.shape)   # (48, 49, 48) (1, 10) () torch.Size([10, 10])
#            fhx = -mu * G_numpy.T.dot(hGx2)
            #print(fhx.shape)    #torch.Size([242, 43])
#            dxh = np.dot(dxh, fhx)
#            print(dxh.shape)    #(242, 43)
#            dxG = np.zeros((n_var, rowSizeG, n_var))
#            for i in range(n_var):
#                dxG[..., i] = np.dot(dxGTemp, fGx[..., i])
            dxG = np.dot(dxGTemp, fGx)
#            print(dxG.shape)    # (48, 49, 48)
#            time_end3=time.time()
#            print('Compute dxG cost:',time_end3-time_end2,'s')
            
            # need newA, newb, newX
#            print(A.shape, b.shape, c.shape)    # torch.Size([18, 31]) torch.Size([18]) torch.Size([31])
#            print(A,b,c)
#            newA = torch.zeros((rowSizeG+2*colSizeG, 4*colSizeG))   # (p+2d)*4d
#            newb = torch.zeros(rowSizeG+2*colSizeG) # p+2d
#            newX = torch.zeros(4*colSizeG)
#            print(newA.shape, newb.shape, newX.shape)   # torch.Size([44, 52]) torch.Size([44]) torch.Size([52])
            A = A[:, :colSizeG]
#            print(A.shape) # torch.Size([18, 13])
            newX = torch.cat((x_sol, s_sol, t_sol), 0)
#            print(newX)
            rowSizeG = A.size(dim=0)
            zeroMatrix = torch.zeros((rowSizeG, 3*colSizeG))
            newA1 = torch.cat((A, zeroMatrix), 1)
            identityMatrix = torch.eye(colSizeG)
            zeroMatrix1 = torch.zeros((colSizeG, colSizeG))
            newA2 = torch.cat((identityMatrix, identityMatrix, zeroMatrix1, -1*identityMatrix), 1)
            newA3 = torch.cat((-1*identityMatrix, zeroMatrix1, identityMatrix, -1*identityMatrix), 1)
            newA = torch.cat((newA1, newA2, newA3), 0)
            newb = torch.cat((b, x_s1, -x_s1), 0)
            newA = newA.numpy()
            newb = newb.numpy()
            newX = newX.numpy()
            newH = np.zeros((4*colSizeG, 4*colSizeG))
#            print(newX)
            for i in range(4*colSizeG):
                if newX[i] != 0:
                    newH[i][i] = mu / (newX[i]*newX[i])
                else:
                    newH[i][i] = 0
            if np.linalg.det(newH) == 0:
                newH = newH + damping*np.eye(4*colSizeG)
            newH = np.linalg.inv(newH)
            newAHA = newA.dot(newH).dot(newA.T)
#            print("1:", np.linalg.det(newAHA))
#            if np.linalg.det(newAHA) == 0:
#                newAHA = newAHA + damping*np.eye(rowSizeG+2*colSizeG)
#                print("2:", np.linalg.det(newAHA))
#            print(newAHA.shape)
#            newAHA.replace([np.inf,-np.inf],0)
            # for i in range(rowSizeG):
            #   for j in range(2*colSizeG)：
            #     if newAHA[i][j] == np.inf or newAHA[i][j] == -np.inf:
            #       newAHA[i][j] = 0
            # newAHA[newAHA == np.inf] = 0
            # newAHA[newAHA == -np.inf] = 0
            try:
              newAHA_inverse = np.linalg.pinv(newAHA)
            except:
              newAHA_inverse = np.eye(rowSizeG+2*colSizeG)
            identityMatrix1 = np.eye(rowSizeG+2*colSizeG)
            dx2b = newH.dot(newA.T).dot(newAHA_inverse).dot(identityMatrix1)
#            print(dxs2b.shape)    # (52, 44)
            zeroMatrix2 = np.zeros((rowSizeG, colSizeG))
            dbxs1 = np.concatenate((zeroMatrix2, identityMatrix, -1*identityMatrix), 0)
#            print(dbxs1.shape)  # (44, 13)
            zeroMatrix3 = np.zeros((colSizeG, 3*colSizeG))
            dxs2x = np.concatenate((identityMatrix, zeroMatrix3), 1)
#            print(dxs2x.shape)  # (13, 52)
            dxs2xs1 = dxs2x.dot(dx2b).dot(dbxs1)
#            print(dxs2xs1.shape)    # (13, 13)
            
#            print(cTrue)
            dRxs2 = np.zeros(colSizeG)
            dRxs1 = np.zeros(colSizeG)
            for i in range(colSizeG):
                dRxs2[i] = purchase_fee * cTrue[i] + (compensation_fee - purchase_fee) * cTrue[i]
                dRxs1[i] = -(compensation_fee - purchase_fee) * cTrue[i]
#                if x_sol[i] > x_s1[i]:
#                    dRxs2[i] = cTrue[i] + penalty[i] * cTrue[i]
#                    dRxs1[i] = -penalty[i] * cTrue[i]
#                else:
#                    dRxs2[i] = cTrue[i] - penalty[i] * cTrue[i]
#                    dRxs1[i] = penalty[i] * cTrue[i]
            
#            print(dRxs2.shape, dxs2xs1.shape, dxh.shape, dRxs1.shape)   # (13,) (13, 13) (13, 18) (13,)
#            dRxs2 = dRxs2 * del_x.numpy()
            
            dxG = dRxs2.dot(dxs2xs1).dot(dxG) + dRxs1.dot(dxG)
            dxc_numpy = dRxs2.dot(dxs2xs1).dot(dxc_numpy) + dRxs1.dot(dxc_numpy)
#            for i in range(colSizeG):
#                dRxs1[i] = cTrue[i]
#            dxc_numpy = dRxs1.dot(dxc_numpy)
#                for i in range(n_var):
#                    dxG[..., i] = np.dot(corrGrad, dxG[..., i])
            
            dxG = torch.from_numpy(dxG)
#            dxG = dxG.type(torch.FloatTensor)
            dxc = torch.from_numpy(dxc_numpy)
#            dxc = dxc.type(torch.FloatTensor)
#            print(dxG.shape)    # torch.Size([48, 49, 48])
#            time_end4=time.time()
#            print('Compute dxG and corrGrad cost:',time_end4-time_end3,'s')
            
            #print("shape:: dxc  {}".format(dxc.shape))  # shape:: dxc  torch.Size([50, 50])
            #dxc_cut = dxc[0:n_var,0:n_var]
            #print(n_var) # 48
            #print("shape:: dxc_cut  {} del_x  {}".format(dxc_cut.shape, del_x.shape))
            # shape:: dxc_cut  torch.Size([48, 48]) del_x  torch.Size([48])
            #print("shape:: del_x  {}".format(del_x.shape)) #torch.Size([242])
            #print(x)
#            print(dxh.shape, del_x.shape)   #torch.Size([242, 43]) torch.Size([242])
            #dh = torch.mv(dxh, del_x)
#            dh = torch.matmul(del_x, dxh)
#            print(del_x.shape, dxG.shape)
            dc = torch.matmul(dxc, del_x)
            dG = torch.matmul(dxG, del_x)
#            print(dG.shape)
#            for i in range(n_var):
#                dxG[..., i] = np.dot(dxGTemp, fGx[..., i])
            #print("shape:: dh  {}".format(dh.shape))  # shape:: dc  torch.Size([48])
            #print("dc ",dc)
            
#            c_grad = torch.zeros(ctx.n_x,dtype=torch.float)
#            c_grad[ctx.var_index] = dh
            c_grad = dc.T
            G_grad = dG.T
#            print(c_grad.shape) #torch.Size([10])
#            print(c_grad)
#            print("Gradient solved")
#            time_end5=time.time()
#            print('Compute G_grad cost:',time_end5-time_end4,'s')
            cG_grad = torch.zeros((n_var, 2))
            cG_grad[:, 0] = c_grad
            cG_grad[:, 1] = G_grad
#            if any(torch.isnan(c_grad).tolist()):
#                logging.info("nan in bkwd pass ; del_x contains NaN?- {}".format(any(torch.isnan(del_x).tolist())))
#            if any(torch.isinf(c_grad).tolist()):
#                logging.info("Inf in bkwd pass ; del_x contains Inf?- {}".format(any(torch.isinf(del_x).tolist())))
            
            #dc = -del_x*x**2/mu
            #dA = -y*del_x*x**2/mu
            # dG = -z*del_x*x**2/mu
            # dG = torch.unsqueeze(dG, 0)
            # dA = torch.unsqueeze(dA, 0)


            end = time.time()
            run_time += end -start 
            # print("bkwd gradient",c_grad) 

            return cG_grad
    def Runtime():
            return run_time
    def end_vectors():
            return save_for_initialization
    def Runtime():
            return run_time
    def forward_solved():
        return problem_solved


    IPOfunc.Runtime = Runtime
    IPOfunc.end_vectors = end_vectors
    IPOfunc.forward_solved = forward_solved    
    return IPOfunc_cls.apply
    

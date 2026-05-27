"""Solver backend names shared by controller examples."""

SOLVER_CHOICES = ("auto", "scipy_slsqp", "jax_slsqp", "casadi_ipopt")


def normalize_solver(solver, *, default="scipy_slsqp"):
    solver = "auto" if solver is None else str(solver)
    if solver == "auto":
        solver = default
    if solver not in SOLVER_CHOICES:
        choices = "', '".join(SOLVER_CHOICES)
        raise ValueError(f"solver must be one of '{choices}'")
    return solver


def require_solver(solver, supported, *, controller):
    solver = normalize_solver(solver)
    supported = tuple(supported)
    if solver not in supported:
        choices = "', '".join(supported)
        raise NotImplementedError(
            f"{controller} currently supports solver='{solver}' only through "
            f"these backends: '{choices}'."
        )
    return solver

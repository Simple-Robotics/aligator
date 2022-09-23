import proxnlp
import proxddp
import numpy as np
from proxnlp import manifolds, constraints

from proxddp import TrajOptProblem, StageModel


def _split_state_control(space, x, N):
    x2 = x.copy()
    sp_x = space.split(x2).tolist()
    xs = []
    us = []
    for i in range(N):
        xs.append(sp_x[2 * i])
        us.append(sp_x[2 * i + 1])
    xs.append(sp_x[-1])
    return xs, us


class ProxnlpCostFromProblem(proxnlp.costs.CostFunctionBase):
    def __init__(self, problem: TrajOptProblem):
        self.problem = problem
        self.prob_data = proxddp.TrajOptData(problem)
        self.space = _get_product_space(problem)
        super().__init__(self.space.nx, self.space.ndx)

    def call(self, x):
        N = self.problem.num_steps
        xs, us = _split_state_control(self.space, x.copy(), N)
        assert len(xs) == N + 1
        assert len(us) == N
        self.problem.evaluate(xs, us, self.prob_data)
        return proxddp.computeTrajectoryCost(self.problem, self.prob_data)

    def computeGradient(self, x, gout):
        N = self.problem.num_steps
        xs, us = _split_state_control(self.space, x.copy(), N)
        self.problem.computeDerivatives(xs, us, self.prob_data)
        gs = self.space.split_vector(gout)
        for i in range(N):
            sd: proxddp.StageData = self.prob_data.stage_data[i]
            cd: proxddp.CostData = sd.cost_data
            gs[2 * i][:] = cd.Lx
            gs[2 * i + 1][:] = cd.Lu
        tcd: proxddp.CostData = self.prob_data.term_cost
        gs[-1][:] = tcd.Lx
        gs.append(tcd.Lx)

    def computeHessian(self, x, Hout):
        N = self.problem.num_steps
        xs, us = _split_state_control(self.space, x.copy(), N)
        k = 0
        for i in range(N):
            sd: proxddp.StageData = self.prob_data.stage_data[i]
            cd: proxddp.CostData = sd.cost_data
            sm: proxddp.StageModel = self.problem.stages[i]
            nx_u = sm.ndx1 + sm.nu
            xr = slice(k, k + nx_u)
            Hout[xr, xr] = cd.hess
            k += nx_u


def _get_start_end_idx(problem: TrajOptProblem):
    k = 0
    N = problem.num_steps
    st = []
    en = []
    for i in range(N):
        sm: proxddp.StageModel = problem.stages[i]
        nt = sm.ndx1 + sm.nu
        st.append(k)
        en.append(k + nt + sm.ndx2)
        k += nt
    return st, en


class ProxnlpConstraintFromProblem(proxnlp.C2Function):
    def __init__(
        self,
        space: manifolds.ManifoldAbstract,
        func: proxddp.StageFunction,
        i: int,
        N: int,
        start_idx: int,
        end_idx: int,
    ):
        nr = func.nr
        super().__init__(space.nx, space.ndx, nr)
        self.space = space
        self.func = func
        self.data = func.createData()
        self.i = i
        self.N = N
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __call__(self, x):
        xs, us = _split_state_control(self.space, x, self.N)
        i = self.i
        self.func.evaluate(xs[i], us[i], xs[i + 1], self.data)
        return self.data.value.copy()

    def computeJacobian(self, x, Jout):
        Jout[:] = 0.0
        Jout = Jout.reshape((self.nr, self.space.ndx))
        xs, us = _split_state_control(self.space, x, self.N)
        i = self.i
        yid = min(i + 1, self.N)
        self.func.computeJacobians(xs[i], us[i], xs[yid], self.data)
        xr = slice(self.start_idx, self.end_idx)
        Jout[:, xr] = self.data.jac_buffer_


def _get_product_space(problem: TrajOptProblem):
    stages: list[StageModel] = problem.stages.tolist()
    N = problem.num_steps
    product_space: manifolds.CartesianProduct = stages[0].xspace
    for i in range(N):
        product_space = product_space * stages[i].uspace
        product_space = product_space * stages[i].xspace_next

    assert product_space.num_components == 2 * N + 1

    return product_space


def convert_problem_to_proxnlp(problem: TrajOptProblem):
    product_space = _get_product_space(problem)
    N = problem.num_steps
    x = product_space.rand()
    sp_x = product_space.split(x).tolist()
    assert len(sp_x) == (N + 1) + N

    cost = ProxnlpCostFromProblem(problem)
    v = cost.call(x)
    assert v == cost(x)[0]
    g = np.zeros(product_space.ndx)
    cost.computeGradient(x, g)
    H = np.zeros((product_space.ndx, product_space.ndx))
    cost.computeHessian(x, H)

    st_idx, en_idx = _get_start_end_idx(problem)
    prnlp_constraints = []
    for i in range(N):
        sm: proxddp.StageModel = problem.stages[i]
        fun = sm.getConstraint(0).func
        cstr_fun = ProxnlpConstraintFromProblem(
            product_space, fun, i, N, st_idx[i], en_idx[i]
        )
        prnlp_constraints.append(constraints.create_equality_constraint(cstr_fun))

    p2 = proxnlp.Problem(cost, prnlp_constraints)
    print(p2.total_constraint_dim)
    return product_space, p2

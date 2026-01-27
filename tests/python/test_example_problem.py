import aligator
import numpy as np

import pytest

HAS_PINOCCHIO = aligator.has_pinocchio_features()
pytestmark = pytest.mark.skipif(
    not HAS_PINOCCHIO, reason="Aligator was compiled without Pinocchio."
)

np.set_printoptions(precision=4, linewidth=250)


@pytest.mark.parametrize("nsteps", [1, 4])
class TestClass:
    def test_dyn(self, nsteps):
        import example_problem as ep

        dyn_data = ep.dyn_model.createData()
        assert isinstance(dyn_data, ep.TwistData)
        dyn_data.Jx[:, :] = np.arange(ep.ndx**2).reshape(ep.ndx, ep.ndx)
        dyn_data.Ju[:, :] = np.arange(ep.ndx**2, ep.ndx**2 + ep.ndx * ep.nu).reshape(
            ep.ndx, ep.nu
        )
        ep.dyn_model.forward(ep.x0, ep.u0, dyn_data)
        ep.dyn_model.dForward(ep.x0, ep.u0, dyn_data)
        print(ep.stage_model.dynamics)
        assert isinstance(ep.stage_model.dynamics, ep.TwistModelExplicit)

    def test_cost(self, nsteps):
        import example_problem as ep

        cost = ep.cost
        cost_data = cost.createData()
        cost.evaluate(ep.x0, ep.u0, cost_data)
        cost.computeGradients(ep.x0, ep.u0, cost_data)
        cost.computeHessians(ep.x0, ep.u0, cost_data)
        assert isinstance(ep.stage_model.cost, ep.MyQuadCost)

    def test_stage(self, nsteps):
        import example_problem as ep

        stage_model = ep.stage_model
        sd = stage_model.createData()
        stage_model.computeFirstOrderDerivatives(ep.x0, ep.u0, sd)
        stage_model.num_dual == ep.ndx

    def test_rollout(self, nsteps):
        import example_problem as ep

        us_i = [np.ones(ep.dyn_model.nu) * 0.1 for _ in range(nsteps)]
        xs_i = aligator.rollout(ep.dyn_model, ep.x0, us_i).tolist()
        dd = ep.dyn_model.createData()
        assert isinstance(dd, ep.TwistData)
        ep.dyn_model.forward(ep.x0, us_i[0], dd)
        assert np.allclose(dd.xnext, xs_i[1])

    def test_shooting_problem(self, nsteps):
        import example_problem as ep

        stage_model = ep.stage_model
        problem = aligator.TrajOptProblem(ep.x0, ep.nu, ep.space, term_cost=ep.cost)
        for _ in range(nsteps):
            problem.addStage(stage_model)

        problem_data = aligator.TrajOptData(problem)

        print("term cost data:", problem_data.term_cost)
        print("term cstr data:", problem_data.term_constraint)

        stage2 = stage_model.copy()
        sd0 = stage2.createData()
        print("Clone stage:", stage2)
        print("Clone stage data:", sd0)

        us_init = [ep.u0] * nsteps
        xs_out = aligator.rollout(ep.dyn_model, ep.x0, us_init).tolist()

        assert len(problem_data.stage_data) == problem.num_steps
        assert problem.num_steps == nsteps

        problem.evaluate(xs_out, us_init, problem_data)
        problem.computeDerivatives(xs_out, us_init, problem_data)

        solver = ep.solver

        assert solver.bcl_params.prim_alpha == 0.1
        assert solver.bcl_params.prim_beta == 0.9
        assert solver.bcl_params.dual_alpha == 1.0
        assert solver.bcl_params.dual_beta == 1.0

        solver.setup(problem)
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.run(problem, xs_out, us_init)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))

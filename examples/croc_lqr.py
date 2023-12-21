import crocoddyl
import numpy as np
import aligator
from aligator import SolverProxDDP


nx = 3
nu = 3
actionmodel = crocoddyl.ActionModelLQR(nx, nu)

state = crocoddyl.StateVector(nx)
x0 = np.random.randn(nx)
nsteps = 10
ams = [actionmodel] * nsteps
termmodel = actionmodel

pb = crocoddyl.ShootingProblem(x0, ams, termmodel)
solver1 = crocoddyl.SolverFDDP(pb)

xs_i = [x0] * (nsteps + 1)
us_i = pb.quasiStatic(xs_i[:nsteps])

maxiter = 2
flag = solver1.solve(xs_i, us_i, maxiter)
assert flag
print(solver1.stop)

# need 2 iters to compute stopping crit
maxiter = 1
flag = solver1.solve(xs_i, us_i, maxiter)
assert not flag
print(solver1.stop)

TOL = 1e-8


prox_pb = aligator.croc.convertCrocoddylProblem(pb)
solver2 = SolverProxDDP(TOL, 1e-10, verbose=aligator.VERBOSE)
solver2.setup(prox_pb)
flag = solver2.run(prox_pb, xs_i, us_i)
print(flag)
print(solver2.results)
print(solver2.workspace)

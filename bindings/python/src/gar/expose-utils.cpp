#include "aligator/python/fwd.hpp"
#include "aligator/gar/utils.hpp"

#include <eigenpy/std-array.hpp>

namespace aligator::python {
using namespace gar;

using context::Scalar;
using lqr_t = LQRProblemTpl<context::Scalar>;

bp::dict lqr_sol_initialize_wrap(const lqr_t &problem) {
  bp::dict out;
  auto ss = lqrInitializeSolution(problem);
  auto &[xs, us, vs, lbdas] = ss;
  out["xs"] = xs;
  out["us"] = us;
  out["vs"] = vs;
  out["lbdas"] = lbdas;
  return out;
}

void exposeGarUtils() {

  bp::def(
      "lqrDenseMatrix",
      +[](const lqr_t &problem, Scalar mudyn, Scalar mueq) {
        auto mat_rhs = lqrDenseMatrix(problem, mudyn, mueq);
        return bp::make_tuple(std::get<0>(mat_rhs), std::get<1>(mat_rhs));
      },
      ("problem"_a, "mudyn", "mueq"));

  bp::def("lqrCreateSparseMatrix", lqrCreateSparseMatrix<Scalar>,
          ("problem"_a, "mudyn", "mueq", "mat", "rhs", "update"),
          "Create or update a sparse matrix from an LQRProblem.");

  bp::def("lqrInitializeSolution", lqr_sol_initialize_wrap, ("problem"_a));

  bp::def("lqrComputeKktError", lqrComputeKktError<Scalar>,
          ("problem"_a, "xs", "us", "vs", "lbdas", "mudyn", "mueq", "theta",
           "verbose"_a = false),
          "Compute the KKT residual of the LQR problem.");
}
} // namespace aligator::python

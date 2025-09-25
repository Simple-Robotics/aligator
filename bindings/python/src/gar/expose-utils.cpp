#include "aligator/python/fwd.hpp"
#include "aligator/gar/utils.hpp"

#include <eigenpy/std-array.hpp>

namespace aligator::python {
using namespace gar;

using context::Scalar;
using context::VectorXs;
using lqr_t = LqrProblemTpl<context::Scalar>;

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

bp::tuple lqr_create_sparse_wrap(const lqr_t &problem, const Scalar mueq,
                                 bool update) {
  Eigen::SparseMatrix<Scalar> mat;
  VectorXs rhs;
  lqrCreateSparseMatrix(problem, mueq, mat, rhs, update);
  mat.makeCompressed();
  return bp::make_tuple(mat, rhs);
}

void exposeGarUtils() {

  bp::def(
      "lqrDenseMatrix",
      +[](const lqr_t &problem, const Scalar mueq) {
        auto [mat, rhs] = lqrDenseMatrix(problem, mueq);
        return bp::make_tuple(mat, rhs);
      },
      ("problem"_a, "mueq"));

  bp::def("lqrCreateSparseMatrix", lqr_create_sparse_wrap,
          ("problem"_a, "mueq", "update"),
          "Create or update a sparse matrix from an LqrProblem.");

  bp::def("lqrInitializeSolution", lqr_sol_initialize_wrap, ("problem"_a));

  bp::def("lqrComputeKktError", lqrComputeKktError<Scalar>,
          ("problem"_a, "xs", "us", "vs", "lbdas", "mudyn", "mueq", "theta",
           "verbose"_a = false),
          "Compute the KKT residual of the LQR problem.");
}
} // namespace aligator::python

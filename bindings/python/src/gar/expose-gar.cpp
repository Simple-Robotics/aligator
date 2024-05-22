/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/blk-matrix.hpp"
#include "aligator/gar/lqr-problem.hpp"
#include "aligator/gar/riccati-base.hpp"
#include "aligator/gar/utils.hpp"

#include "aligator/python/utils.hpp"
#include "aligator/python/visitors.hpp"

#include <eigenpy/std-array.hpp>

namespace aligator::python {
using namespace gar;

using context::Scalar;
using riccati_base_t = RiccatiSolverBase<Scalar>;
using knot_t = LQRKnotTpl<context::Scalar>;
using lqr_t = LQRProblemTpl<context::Scalar>;

using context::MatrixXs;
using RowMatrixXs = Eigen::Transpose<MatrixXs>::PlainMatrix;
using context::VectorXs;

using knot_vec_t = lqr_t::KnotVector;

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

static void exposeBlockMatrices() {
  BlkMatrixPythonVisitor<BlkMatrix<MatrixXs, 2, 2>>::expose("BlockMatrix22");
  BlkMatrixPythonVisitor<BlkMatrix<VectorXs, 4, 1>>::expose("BlockVector4");
  BlkMatrixPythonVisitor<BlkMatrix<VectorXs, 2, 1>>::expose("BlockVector2");
  BlkMatrixPythonVisitor<BlkMatrix<RowMatrixXs, 4, 1>>::expose(
      "BlockRowMatrix41");
  BlkMatrixPythonVisitor<BlkMatrix<RowMatrixXs, 2, 1>>::expose(
      "BlockRowMatrix21");
  eigenpy::StdArrayPythonVisitor<std::array<long, 1>, true>::expose(
      "StdArr1_long");
  eigenpy::StdArrayPythonVisitor<std::array<long, 2>, true>::expose(
      "StdArr2_long");
  eigenpy::StdArrayPythonVisitor<std::array<long, 4>, true>::expose(
      "StdArr4_long");
}

#ifdef ALIGATOR_WITH_CHOLMOD
// fwd-declare exposeCholmodSolver()
void exposeCholmodSolver();
#endif
// fwd-declare exposeParallelSolver()
void exposeParallelSolver();
// fwd-declare exposeDenseSolver()
void exposeDenseSolver();
// fwd-declare exposeProxRiccati()
void exposeProxRiccati();

void exposeGAR() {

  bp::scope ns = get_namespace("gar");

  exposeBlockMatrices();

  bp::class_<knot_t>("LQRKnot", bp::no_init)
      .def(bp::init<uint, uint, uint>(("nx"_a, "nu", "nc")))
      .def(bp::init<uint, uint, uint, uint>(("nx"_a, "nu", "nc", "nx2")))
      .def_readonly("nx", &knot_t::nx)
      .def_readonly("nu", &knot_t::nu)
      .def_readonly("nc", &knot_t::nc)
      .def_readonly("nx2", &knot_t::nx2)
      .def_readonly("nth", &knot_t::nth)
      //
      .def_readwrite("Q", &knot_t::Q)
      .def_readwrite("S", &knot_t::S)
      .def_readwrite("R", &knot_t::R)
      .def_readwrite("q", &knot_t::q)
      .def_readwrite("r", &knot_t::r)
      //
      .def_readwrite("A", &knot_t::A)
      .def_readwrite("B", &knot_t::B)
      .def_readwrite("E", &knot_t::E)
      .def_readwrite("f", &knot_t::f)
      //
      .def_readwrite("C", &knot_t::C)
      .def_readwrite("D", &knot_t::D)
      .def_readwrite("d", &knot_t::d)
      //
      .def_readwrite("Gth", &knot_t::Gth)
      .def_readwrite("Gx", &knot_t::Gx)
      .def_readwrite("Gu", &knot_t::Gu)
      .def_readwrite("gamma", &knot_t::gamma)
      //
      .def(CopyableVisitor<knot_t>())
      .def(PrintableVisitor<knot_t>());

  StdVectorPythonVisitor<knot_vec_t, false>::expose("StdVec_LQRKnot");

  bp::class_<lqr_t>("LQRProblem", bp::no_init)
      .def(bp::init<const knot_vec_t &, long>(("self"_a, "stages", "nc0")))
      .def_readwrite("stages", &lqr_t::stages)
      .add_property("horizon", &lqr_t::horizon)
      .def_readwrite("G0", &lqr_t::G0)
      .def_readwrite("g0", &lqr_t::g0)
      .add_property("isInitialized", &lqr_t::isInitialized,
                    "Whether the problem is initialized.")
      .add_property("isParameterized", &lqr_t::isParameterized,
                    "Whether the problem is parameterized.")
      .def("addParameterization", &lqr_t::addParameterization,
           ("self"_a, "nth"))
      .add_property("ntheta", &lqr_t::ntheta)
      .def("evaluate", &lqr_t::evaluate,
           ("self"_a, "xs", "us", "theta"_a = std::nullopt),
           "Evaluate the problem objective.")
      .def(CopyableVisitor<lqr_t>());

  bp::class_<riccati_base_t, boost::noncopyable>("RiccatiSolverBase",
                                                 bp::no_init)
      .def("backward", &riccati_base_t::backward, ("self"_a, "mu", "mueq"))
      .def("forward", &riccati_base_t::forward,
           ("self"_a, "xs", "us", "vs", "lbdas", "theta"_a = std::nullopt));

  bp::def(
      "lqrDenseMatrix",
      +[](const lqr_t &problem, Scalar mudyn, Scalar mueq) {
        auto mat_rhs = lqrDenseMatrix(problem, mudyn, mueq);
        return bp::make_tuple(std::get<0>(mat_rhs), std::get<1>(mat_rhs));
      },
      ("problem"_a, "mudyn", "mueq"));

  bp::def("lqrInitializeSolution", lqr_sol_initialize_wrap, ("problem"_a));

#ifdef ALIGATOR_WITH_CHOLMOD
  exposeCholmodSolver();
#endif
  exposeDenseSolver();
  exposeParallelSolver();
  exposeProxRiccati();
}

} // namespace aligator::python

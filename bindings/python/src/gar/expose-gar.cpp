/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/python/gar-visitors.hpp"
#include "aligator/gar/riccati.hpp"
#include "aligator/gar/utils.hpp"

#include "aligator/python/utils.hpp"
#include "aligator/python/visitors.hpp"

namespace aligator {
namespace python {
using namespace gar;

using context::Scalar;
using prox_riccati_t = ProximalRiccatiSolver<Scalar>;
using StageFactor = prox_riccati_t::StageFactorType;
using knot_t = LQRKnotTpl<context::Scalar>;
using lqr_t = LQRProblemTpl<context::Scalar>;
} // namespace python
} // namespace aligator

#if EIGENPY_VERSION_AT_MOST(3, 2, 0)
namespace eigenpy {
namespace internal {
template <>
struct has_operator_equal<::aligator::python::knot_t> : boost::false_type {};
template <>
struct has_operator_equal<::aligator::python::StageFactorType>
    : boost::false_type {};
} // namespace internal
} // namespace eigenpy
#endif

namespace aligator {
namespace python {
using context::MatrixXs;
using RowMatrixXs = Eigen::Transpose<MatrixXs>::PlainMatrix;
using context::VectorXs;

using knot_vec_t = std::vector<knot_t>;

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
}

// fwd-declare exposeParallelSolver()
void exposeParallelSolver();

void exposeGAR() {

  bp::scope ns = get_namespace("gar");

  exposeBlockMatrices();

  using value_t = prox_riccati_t::value_t;
  bp::class_<value_t>("value_data", bp::no_init)
      .def_readonly("Pmat", &value_t::Pmat)
      .def_readonly("pvec", &value_t::pvec)
      .def_readonly("Vxx", &value_t::Vxx)
      .def_readonly("vx", &value_t::vx)
      .def_readonly("Vxt", &value_t::Vxt)
      .def_readonly("Vtt", &value_t::Vtt)
      .def_readonly("vt", &value_t::vt);

  bp::class_<StageFactor>(
      "StageFactor", "Stagewise factor for the generalized Riccati algorithm.",
      bp::no_init)
      .def_readonly("ff", &StageFactor::ff)
      .def_readonly("fb", &StageFactor::fb)
      .def_readonly("fth", &StageFactor::fth)
      .def_readonly("kktMat", &StageFactor::kktMat)
      .def_readonly("kktChol", &StageFactor::kktChol)
      .def_readonly("vm", &StageFactor::vm);

  StdVectorPythonVisitor<std::vector<StageFactor>, true>::expose(
      "StdVec_stage_factor");

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

  {
    bp::scope _ =
        bp::class_<prox_riccati_t, boost::noncopyable>(
            "ProximalRiccatiSolver", "Proximal Riccati solver.", bp::no_init)
            .def(bp::init<const lqr_t &>(("self"_a, "problem")))
            .def_readonly("datas", &prox_riccati_t::datas)
            .def_readonly("thGrad", &prox_riccati_t::thGrad, "Value gradient")
            .def_readonly("thHess", &prox_riccati_t::thHess, "Value Hessian")
            .def_readonly("kkt0", &prox_riccati_t::kkt0,
                          "Initial stage KKT system")
            .def("backward", &prox_riccati_t::backward,
                 ("self"_a, "mu", "mueq"))
            .def("forward", &prox_riccati_t::forward,
                 ("self"_a, "xs", "us", "vs", "lbdas",
                  "theta"_a = std::nullopt));
    bp::class_<prox_riccati_t::kkt0_t>("kkt0_t", bp::no_init)
        .def_readonly("ff", &prox_riccati_t::kkt0_t::ff)
        .def_readonly("fth", &prox_riccati_t::kkt0_t::fth)
        .def_readonly("mat", &prox_riccati_t::kkt0_t::mat)
        .def_readonly("chol", &prox_riccati_t::kkt0_t::chol);
  }

  bp::def(
      "lqrDenseMatrix",
      +[](const lqr_t &problem, Scalar mudyn, Scalar mueq) {
        auto mat_rhs = lqrDenseMatrix(problem, mudyn, mueq);
        return bp::make_tuple(std::get<0>(mat_rhs), std::get<1>(mat_rhs));
      },
      ("problem"_a, "mudyn", "mueq"));

  bp::def("lqrInitializeSolution", lqr_sol_initialize_wrap, ("problem"_a));

  exposeParallelSolver();
}

} // namespace python
} // namespace aligator

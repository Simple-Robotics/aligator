/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/gar/riccati.hpp"

#include "aligator/python/utils.hpp"
#include "aligator/python/visitors.hpp"

namespace aligator {
namespace python {
using namespace gar;

using riccati_bwd_t = ProximalRiccatiSolverBackward<context::Scalar>;
using riccati_fwd_t = ProximalRiccatiSolverForward<context::Scalar>;
using knot_t = LQRKnot<context::Scalar>;
using stage_solve_data_t = riccati_bwd_t::stage_solve_data_t;
} // namespace python
} // namespace aligator

namespace eigenpy {
namespace internal {
template <>
struct has_operator_equal<::aligator::python::knot_t> : boost::false_type {};
template <>
struct has_operator_equal<::aligator::python::stage_solve_data_t>
    : boost::false_type {};
} // namespace internal
} // namespace eigenpy

namespace aligator {
namespace python {

using knot_vec_t = std::vector<knot_t>;

void exposeGAR() {

  bp::scope ns = get_namespace("gar");

  bp::class_<riccati_bwd_t::value_t>("value_data", bp::no_init)
      .def_readonly("Pmat", &riccati_bwd_t::value_t::Pmat)
      .def_readonly("pvec", &riccati_bwd_t::value_t::pvec)
      .def_readonly("Vmat", &riccati_bwd_t::value_t::Vmat)
      .def_readonly("vvec", &riccati_bwd_t::value_t::vvec);

  bp::class_<riccati_bwd_t::kkt_t>("kkt_data", bp::no_init)
      .def_readonly("data", &riccati_bwd_t::kkt_t::data)
      .def_readonly("chol", &riccati_bwd_t::kkt_t::chol)
      .add_property("R", &riccati_bwd_t::kkt_t::R)
      .add_property("D", &riccati_bwd_t::kkt_t::D);

  bp::class_<stage_solve_data_t>("stage_solve_data", bp::no_init)
      .def_readonly("ff", &stage_solve_data_t::ff)
      .def_readonly("fb", &stage_solve_data_t::fb)
      .def_readonly("Mmat", &stage_solve_data_t::Mmat)
      .def_readonly("vm", &stage_solve_data_t::vm);

  StdVectorPythonVisitor<std::vector<stage_solve_data_t>, true>::expose(
      "stage_solve_data_Vec");

  bp::class_<knot_t>("LQRKnot", bp::no_init)
      .def(bp::init<uint, uint, uint>(bp::args("nx", "nu", "nc")))
      .def_readonly("nx", &knot_t::nx)
      .def_readonly("nu", &knot_t::nu)
      .def_readonly("nc", &knot_t::nc)
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
      .def(CopyableVisitor<knot_t>());

  StdVectorPythonVisitor<knot_vec_t, true>::expose("LQRKnotVec");

  using context::Scalar;
  bp::class_<riccati_bwd_t, boost::noncopyable>(
      "ProximalRiccatiBwd", "Proximal Riccati backward pass.", bp::no_init)
      .def(bp::init<const knot_vec_t &>(bp::args("self", "knots")))
      .add_property("horizon", &riccati_bwd_t::horizon)
      .def_readonly("datas", &riccati_bwd_t::datas)
      .def("run", &riccati_bwd_t::run, bp::args("self", "mu", "mueq"));

  bp::class_<riccati_fwd_t>("ProximalRiccatiFwd",
                            "Proximal Riccati forward pass.", bp::no_init)
      .def("run", &riccati_fwd_t::run,
           bp::args("bwd_pass", "xs", "us", "vs", "lbdas"))
      .staticmethod("run");

  bp::def(
      "lqrDenseMatrix",
      +[](const knot_vec_t &knots, Scalar mudyn, Scalar mueq) {
        auto mat_rhs = lqrDenseMatrix(knots, mudyn, mueq);
        return bp::make_tuple(std::get<0>(mat_rhs), std::get<1>(mat_rhs));
      },
      bp::args("self", "mudyn", "mueq"));
}

} // namespace python
} // namespace aligator

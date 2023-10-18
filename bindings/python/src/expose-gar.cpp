/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/gar/riccati.hpp"

#include "aligator/python/utils.hpp"
#include "aligator/python/visitors.hpp"

namespace aligator {
namespace python {
using namespace gar;

using prox_riccati_t = ProximalRiccatiSolver<context::Scalar>;
using knot_t = LQRKnot<context::Scalar>;
using stage_solve_data_t = prox_riccati_t::stage_solve_data_t;
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

template <typename BlockMatrixType>
struct BlkMatrixPythonVisitor
    : bp::def_visitor<BlkMatrixPythonVisitor<BlockMatrixType>> {

  enum { N = BlockMatrixType::N, M = BlockMatrixType::M };
  using MatrixType = typename BlockMatrixType::MatrixType;

  using Self = BlkMatrixPythonVisitor<BlockMatrixType>;
  template <size_t k> using long_arr = std::array<long, k>;

  static context::MatrixRef get_block(BlockMatrixType &bmt, size_t i,
                                      size_t j) {
    return bmt(i, j);
  }

  static auto ctor1(const std::vector<long> &_dims) {
    long_arr<M> dims;
    std::copy_n(_dims.begin(), M, dims.begin());
    return std::make_shared<BlockMatrixType>(dims, dims);
  }

  static auto ctor2(const std::vector<long> &_rowDims,
                    const std::vector<long> &_colDims) {
    long_arr<N> rowDims;
    std::copy_n(_rowDims.begin(), N, rowDims.begin());
    long_arr<M> colDims;
    std::copy_n(_colDims.begin(), M, colDims.begin());
    return std::make_shared<BlockMatrixType>(rowDims, colDims);
  }

  template <size_t k> static auto stdArrToList(const std::array<long, k> &s) {
    bp::list out;
    for (size_t i = 0; i < s.size(); i++)
      out.append(s[i]);
    return out;
  }

  static auto rowDims(const BlockMatrixType &mat) {
    return stdArrToList(mat.rowDims());
  }
  static auto colDims(const BlockMatrixType &mat) {
    return stdArrToList(mat.colDims());
  }

  template <class... Args> void visit(bp::class_<Args...> &obj) const {
    obj.def("__init__",
            bp::make_constructor(&ctor2, bp::default_call_policies(),
                                 bp::args("rowDims", "colDims")))
        .def_readwrite("data", &BlockMatrixType::data)
        .add_property("rowDims", rowDims)
        .add_property("colDims", colDims)
        .def("__call__", get_block, bp::args("self", "i", "j"));
    if constexpr (N == M) {
      obj.def("__init__",
              bp::make_constructor(&ctor1, bp::default_call_policies(),
                                   bp::args("dims")));
    }
  }

  static void expose(const char *name) {
    bp::class_<BlockMatrixType>(name, "", bp::no_init).def(Self());
  }
};

using knot_vec_t = std::vector<knot_t>;

void exposeGAR() {

  bp::scope ns = get_namespace("gar");

  StdVectorPythonVisitor<std::vector<long>, true>::expose("StdVec_long");

  using BMT22 = BlkMatrix<context::MatrixXs, 2, 2>;
  using BMT21 = BlkMatrix<context::MatrixXs, 2, 1>;
  using BVT2 = BlkMatrix<context::VectorXs, 2, 1>;

  BlkMatrixPythonVisitor<BMT22>::expose("BlockMatrix22");
  BlkMatrixPythonVisitor<BMT21>::expose("BlockMatrix21");
  BlkMatrixPythonVisitor<BVT2>::expose("BlockVector2");

  bp::class_<prox_riccati_t::value_t>("value_data", bp::no_init)
      .def_readonly("Pmat", &prox_riccati_t::value_t::Pmat)
      .def_readonly("pvec", &prox_riccati_t::value_t::pvec)
      .def_readonly("Vmat", &prox_riccati_t::value_t::Vmat)
      .def_readonly("vvec", &prox_riccati_t::value_t::vvec);

  bp::class_<prox_riccati_t::kkt_t, bp::bases<BMT22>>("kkt_data", bp::no_init)
      .def_readonly("data", &prox_riccati_t::kkt_t::data)
      .def_readonly("chol", &prox_riccati_t::kkt_t::chol)
      .add_property("R", &prox_riccati_t::kkt_t::R)
      .add_property("D", &prox_riccati_t::kkt_t::D);

  bp::class_<stage_solve_data_t>("stage_solve_data", bp::no_init)
      .def_readonly("ff", &stage_solve_data_t::ff)
      .def_readonly("fb", &stage_solve_data_t::fb)
      .def_readonly("kkt", &stage_solve_data_t::kkt)
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
      .def(CopyableVisitor<knot_t>())
      .def(PrintableVisitor<knot_t>());

  StdVectorPythonVisitor<knot_vec_t, true>::expose("LQRKnotVec");

  using context::Scalar;
  bp::class_<prox_riccati_t, boost::noncopyable>(
      "ProximalRiccatiBwd", "Proximal Riccati backward pass.", bp::no_init)
      .def(bp::init<const knot_vec_t &>(bp::args("self", "knots")))
      .add_property("horizon", &prox_riccati_t::horizon)
      .def_readonly("datas", &prox_riccati_t::datas)
      .def("backward", &prox_riccati_t::backward,
           bp::args("self", "mu", "mueq"))
      .def("forward", &prox_riccati_t::forward,
           bp::args("self", "xs", "us", "vs", "lbdas"));

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

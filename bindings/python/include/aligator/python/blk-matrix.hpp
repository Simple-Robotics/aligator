#include "fwd.hpp"
#include "aligator/core/blk-matrix.hpp"

namespace aligator {
namespace python {
namespace bp = boost::python;

template <typename BlockMatrixType> struct BlkMatrixPythonVisitor;

template <typename MatrixType, int N, int M>
struct BlkMatrixPythonVisitor<BlkMatrix<MatrixType, N, M>>
    : bp::def_visitor<BlkMatrixPythonVisitor<BlkMatrix<MatrixType, N, M>>> {
  using BlockMatrixType = BlkMatrix<MatrixType, N, M>;
  using RefType = Eigen::Ref<MatrixType>;

  using Self = BlkMatrixPythonVisitor<BlockMatrixType>;

  static RefType get_block(BlockMatrixType &bmt, size_t i, size_t j) {
    return bmt(i, j);
  }

  static RefType blockRow(BlockMatrixType &mat, size_t i) {
    if (i >= mat.rowDims().size()) {
      PyErr_SetString(PyExc_IndexError, "Index out of range.");
      bp::throw_error_already_set();
    }
    return mat.blockRow(i);
  }

  template <class... Args> void visit(bp::class_<Args...> &obj) const {
    obj.add_property(
           "matrix", +[](BlockMatrixType &m) -> RefType { return m.matrix(); })
        .def_readonly("rows", &BlockMatrixType::rows)
        .def_readonly("cols", &BlockMatrixType::cols)
        .add_property("rowDims",
                      bp::make_function(&BlockMatrixType::rowDims,
                                        bp::return_internal_reference<>()))
        .add_property("colDims",
                      bp::make_function(&BlockMatrixType::colDims,
                                        bp::return_internal_reference<>()))
        .def("blockRow", blockRow, "Get a block row by index.")
        .def("__call__", get_block, ("self"_a, "i", "j"));
  }

  static void expose(const char *name) {
    bp::class_<BlockMatrixType>(name, "", bp::no_init).def(Self());
  }
};

} // namespace python
} // namespace aligator

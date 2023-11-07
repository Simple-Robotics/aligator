#include "fwd.hpp"
#include "aligator/gar/BlkMatrix.hpp"

namespace aligator {
namespace python {
namespace bp = boost::python;

template <typename BlockMatrixType>
struct BlkMatrixPythonVisitor
    : bp::def_visitor<BlkMatrixPythonVisitor<BlockMatrixType>> {

  enum { N = BlockMatrixType::N, M = BlockMatrixType::M };
  using MatrixType = typename BlockMatrixType::MatrixType;
  using RefType = Eigen::Ref<MatrixType>;

  using Self = BlkMatrixPythonVisitor<BlockMatrixType>;
  template <size_t k> using long_arr = std::array<long, k>;

  static RefType get_block(BlockMatrixType &bmt, size_t i, size_t j) {
    return bmt(i, j);
  }

  template <size_t k> static auto stdArrToList(const std::array<long, k> &s) {
    bp::list out;
    for (size_t i = 0; i < s.size(); i++)
      out.append(s[i]);
    return out;
  }

  static auto rowDims(const BlockMatrixType &mat) {
    if constexpr (N == -1)
      return mat.rowDims();
    else
      return stdArrToList(mat.rowDims());
  }
  static auto colDims(const BlockMatrixType &mat) {
    if constexpr (M == -1)
      return mat.colDims();
    else
      return stdArrToList(mat.colDims());
  }

  template <class... Args> void visit(bp::class_<Args...> &obj) const {
    obj.def_readwrite("data", &BlockMatrixType::data)
        .add_property("rowDims", rowDims)
        .add_property("colDims", colDims)
        .def(
            "blockRow",
            +[](BlockMatrixType &mat, size_t i) -> RefType {
              return mat.blockRow(i);
            })
        .def("__call__", get_block, bp::args("self", "i", "j"));
  }

  static void expose(const char *name) {
    bp::class_<BlockMatrixType>(name, "", bp::no_init).def(Self());
  }
};

} // namespace python
} // namespace aligator

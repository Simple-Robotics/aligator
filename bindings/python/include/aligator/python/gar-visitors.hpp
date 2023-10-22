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

} // namespace python
} // namespace aligator

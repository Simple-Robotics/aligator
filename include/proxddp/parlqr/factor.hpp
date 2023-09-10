#include "proxddp/math.hpp"

namespace proxddp {

/// A factor can represent the problem right-hand side data,
/// the problem matrix data, or parts of the matrix (e.g. the Jacobians)
/// involved in its factorizaton.
/// The data is stored in column-major fashion.
template <typename T> struct LQRFactor {
  using RowMat = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;

  LQRFactor(long nx, long nu, long nc, long width)
      : data(nx * 2 + nu + nc, width),
        //
        X(data.topRows(nx)),
        //
        U(data.middleRows(nx, nu)),
        //
        Nu(data.middleRows(nx + nu, nc)),
        //
        Lambda(data.bottomRows(nx)) {
    data.setZero();
  }

  LQRFactor(const LQRFactor &other)
      : LQRFactor(other.X.rows(), other.U.rows(), other.Nu.rows(),
                  other.X.cols()) {
    data = other.data;
  }

  RowMat data;
  // state-input variables
  Eigen::Ref<RowMat> X;
  Eigen::Ref<RowMat> U;
  // dual variables (co-state & constraint multiplier)
  Eigen::Ref<RowMat> Nu;
  Eigen::Ref<RowMat> Lambda;
};

} // namespace proxddp

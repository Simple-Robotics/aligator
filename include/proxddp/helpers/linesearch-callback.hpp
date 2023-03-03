#include "proxddp/core/callback-base.hpp"

namespace proxddp {

template <typename T> struct DataLinesearchDebug {
  PROXNLP_DYNAMIC_TYPEDEFS(T);
  static constexpr int DYN = Eigen::Dynamic;
  using Matrix2Xs = Eigen::Matrix<T, 2, DYN>;

  DataLinesearchDebug() : data_() { clear(); }

  const Matrix2Xs &data() const { return data_; }

  void push(T alpha, T phi) {
    Eigen::Index nc = data_.cols();
    // increase size
    data_.conservativeResize(2, nc + 1);
    data_.col(nc) << alpha, phi;
  }

  void clear() { data_.resize(2, 0); }

private:
  Matrix2Xs data_;
};

template <typename Scalar>
struct LinesearchCallback : helpers::base_callback<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  using Data = DataLinesearchDebug<Scalar>;

  LinesearchCallback() : storage_(), alpha_min(0.), alpha_max(1.) {}

  void post_linesearch_call(boost::any data) override {
    storage_.push_back(boost::any_cast<Data &&>(data));
  }

  void call(const Workspace &, const Results &) override {}

  typename Data::Matrix2Xs const &get(std::size_t t) const {
    return storage_[t].data();
  }

  std::vector<Data> storage_;
  Scalar alpha_min;
  Scalar alpha_max;
};

} // namespace proxddp

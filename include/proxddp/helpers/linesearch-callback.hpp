#pragma once

#include "proxddp/core/callback-base.hpp"

namespace proxddp {
namespace helpers {

template <typename Scalar>
struct LinesearchCallback : helpers::base_callback<Scalar> {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  using InputType = std::tuple<std::function<Scalar(Scalar)>, Scalar, Scalar>;

  struct Data {
    static constexpr int DYN = Eigen::Dynamic;
    using Matrix2Xs = Eigen::Matrix<Scalar, 2, DYN>;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    Data() : data_() { clear(); }

    void push(Scalar alpha, Scalar phi) {
      Eigen::Index nc = data_.cols();
      // increase size
      data_.conservativeResize(2, nc + 1);
      data_.col(nc) << alpha, phi;
    }

    void clear() { data_.resize(2, 0); }

  private:
    Scalar dphi_;
    Matrix2Xs data_;
    friend LinesearchCallback;
  };

  LinesearchCallback() : storage_(), alpha_min(0.), alpha_max(1.) {}

  void post_linesearch_call(boost::any in) override {
    fmt::print("[Found a linesearch debug callback]\n");
    Data d;
    std::function<Scalar(Scalar)> callable;
    Scalar alpha_opt;
    std::tie(callable, d.dphi_, alpha_opt) = boost::any_cast<InputType>(in);

    Scalar a = alpha_min;
    while (a <= alpha_max) {
      Scalar phi = callable(a);
      d.push(a, phi);
      a += da;
    }
    d.push(alpha_opt, callable(alpha_opt));

    storage_.push_back(d);
  }

  void call(const Workspace &, const Results &) override {}

  typename Data::Matrix2Xs const &get(std::size_t t) const {
    return storage_[t].data_;
  }

  Scalar get_dphi(std::size_t t) const { return storage_[t].dphi_; }

  std::vector<Data> storage_;
  Scalar alpha_min;
  Scalar alpha_max;
  Scalar da = 0.01;
};

} // namespace helpers
} // namespace proxddp

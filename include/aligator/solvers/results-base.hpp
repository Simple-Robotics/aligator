/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

template <typename _Scalar> struct ResultsBaseTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

protected:
  // Whether the results struct was initialized.
  bool m_isInitialized;

public:
  std::size_t num_iters = 0;
  bool conv = false;

  Scalar traj_cost_ = 0.;
  Scalar merit_value_ = 0.;
  /// Overall primal infeasibility measure/constraint violation.
  Scalar prim_infeas = 0.;
  /// Overall dual infeasibility measure.
  Scalar dual_infeas = 0.;

  /// Riccati gains
  std::vector<MatrixXs> gains_;
  /// States
  std::vector<VectorXs> xs;
  /// Controls
  std::vector<VectorXs> us;

  ResultsBaseTpl()
      : m_isInitialized(false) {}
  bool isInitialized() const { return m_isInitialized; }

  /// @brief Get column expression of the primal-dual feedforward gain.
  decltype(auto) getFeedforward(std::size_t i) {
    return this->gains_[i].col(0);
  }

  /// @copybrief getFeedforward()
  decltype(auto) getFeedforward(std::size_t i) const {
    return this->gains_[i].col(0);
  }

  /// @brief Get expression of the primal-dual feedback gains.
  decltype(auto) getFeedback(std::size_t i) {
    return this->gains_[i].rightCols(this->get_ndx1(i));
  }

  /// @copybrief getFeedback()
  decltype(auto) getFeedback(std::size_t i) const {
    return this->gains_[i].rightCols(this->get_ndx1(i));
  }

  std::vector<MatrixXs> getCtrlFeedbacks() const {
    const std::size_t N = us.size();
    std::vector<MatrixXs> out;
    out.reserve(N);
    for (std::size_t i = 0; i < N; i++) {
      const Eigen::Index nu = us[i].rows();
      out.emplace_back(getFeedback(i).topRows(nu));
    }
    return out;
  }

  std::vector<VectorXs> getCtrlFeedforwards() const {
    const std::size_t N = us.size();
    std::vector<VectorXs> out;
    out.reserve(N);
    for (std::size_t i = 0; i < N; i++) {
      const Eigen::Index nu = us[i].rows();
      out.emplace_back(getFeedforward(i).head(nu));
    }
    return out;
  }

  std::string printBase() const;
  virtual ~ResultsBaseTpl() = default;

private:
  Eigen::Index get_ndx1(std::size_t i) const {
    return this->gains_[i].cols() - 1;
  }
};

template <typename Scalar>
std::string ResultsBaseTpl<Scalar>::printBase() const {
  return fmt::format("\n  num_iters:    {:d},"
                     "\n  converged:    {},"
                     "\n  traj. cost:   {:.3e},"
                     "\n  merit.value:  {:.3e},"
                     "\n  prim_infeas:  {:.3e},"
                     "\n  dual_infeas:  {:.3e},",
                     num_iters, conv, traj_cost_, merit_value_, prim_infeas,
                     dual_infeas);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss,
                         const ResultsBaseTpl<Scalar> &self) {
  return oss << "Results {" << self.printBase() << "\n}";
}

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "results-base.txx"
#endif

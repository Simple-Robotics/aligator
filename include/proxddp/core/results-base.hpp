/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"

namespace aligator {

template <typename _Scalar> struct ResultsBaseTpl {
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);

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
  /// Problem Lagrange multipliers
  std::vector<VectorXs> lams;

  ResultsBaseTpl() : m_isInitialized(false) {}
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
    for (std::size_t i = 0; i < N; i++) {
      const Eigen::Index nu = us[i].rows();
      out.emplace_back(getFeedback(i).topRows(nu));
    }
    return out;
  }

  std::vector<VectorXs> getCtrlFeedforwards() const {
    const std::size_t N = us.size();
    std::vector<VectorXs> out;
    for (std::size_t i = 0; i < N; i++) {
      const Eigen::Index nu = us[i].rows();
      out.emplace_back(getFeedforward(i).head(nu));
    }
    return out;
  }

  void printBase(std::ostream &oss) const;

private:
  Eigen::Index get_ndx1(std::size_t i) const {
    return this->gains_[i].cols() - 1;
  }
};

template <typename Scalar>
void ResultsBaseTpl<Scalar>::printBase(std::ostream &oss) const {
  oss << fmt::format("\n  num_iters:    {:d},", num_iters)
      << fmt::format("\n  converged:    {},", conv)
      << fmt::format("\n  traj. cost:   {:.3e},", traj_cost_)
      << fmt::format("\n  merit.value:  {:.3e},", merit_value_)
      << fmt::format("\n  prim_infeas:  {:.3e},", prim_infeas)
      << fmt::format("\n  dual_infeas:  {:.3e},", dual_infeas);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss,
                         const ResultsBaseTpl<Scalar> &self) {
  oss << "Results {";
  self.printBase(oss);
  oss << "\n}";
  return oss;
}

} // namespace aligator

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./results-base.txx"
#endif

#pragma once

#include "proxddp/core/shooting-problem.hpp"

#include <fmt/format.h>
#include <ostream>


namespace proxddp
{
  
  /// @brief    Results holder struct.
  template<typename _Scalar>
  struct ResultsTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    std::size_t num_iters = 0;

    /// States
    std::vector<VectorXs> xs_;
    /// Controls
    std::vector<VectorXs> us_;
    /// Problem Lagrange multipliers
    std::vector<VectorXs> lams_;
    /// Dynamics' co-states
    std::vector<VectorRef> co_state_;

    Scalar traj_cost_ = 0.;

    /// @brief    Create the results struct from a problem (ShootingProblemTpl) instance.
    explicit ResultsTpl(const ShootingProblemTpl<Scalar>& problem);

    friend std::ostream& operator<<(std::ostream& oss, const ResultsTpl& self)
    {
      oss << "Results {";
      oss << fmt::format("\n\tnumiters  :  {:d}", self.num_iters);
      oss << ",";
      oss << fmt::format("\n\ttraj. cost:  {:.3e}", self.traj_cost_);
      oss << "\n}";
      return oss;
    }

  };

} // namespace proxddp

#include "proxddp/core/solver-results.hxx"

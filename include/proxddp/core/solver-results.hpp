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
    bool conv = false;

    /// Riccati gains
    std::vector<MatrixXs> gains_;
    /// States
    std::vector<VectorXs> xs_;
    /// Controls
    std::vector<VectorXs> us_;
    /// Problem Lagrange multipliers
    std::vector<VectorXs> lams_;
    /// Dynamics' co-states
    std::vector<VectorRef> co_state_;

    Scalar traj_cost_ = 0.;
    Scalar merit_value_ = 0.;
    /// Overall primal infeasibility/constraint violation for the ShootingProblemTpl.
    Scalar primal_infeasibility = 0.;
    /// Overall dual infeasibility measure for the ShootingProblemTpl.
    Scalar dual_infeasibility = 0.;

    /// @brief    Create the results struct from a problem (ShootingProblemTpl) instance.
    explicit ResultsTpl(const ShootingProblemTpl<Scalar>& problem);

    friend std::ostream& operator<<(std::ostream& oss, const ResultsTpl& self)
    {
      oss << "Results {";
      oss << fmt::format("\n  numiters   :  {:d},", self.num_iters);
      oss << fmt::format("\n  converged  :  {},", self.conv);
      oss << fmt::format("\n  traj. cost :  {:.3e},", self.traj_cost_)
          << fmt::format("\n  merit.value:  {:.3e},", self.merit_value_)
          << fmt::format("\n  prim_infeas:  {:.3e},", self.primal_infeasibility)
          << fmt::format("\n  dual_infeas:  {:.3e},", self.dual_infeasibility);
      oss << "\n}";
      return oss;
    }

  };

} // namespace proxddp

#include "proxddp/core/solver-results.hxx"

#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"


namespace proxddp
{
  /// @brief    Shooting problem, consisting in a succession of nodes.
  ///
  /// @details  The problem can be written as a nonlinear program:
  /// \f[
  ///   \begin{aligned}
  ///     \min_{\bfx,\bfu}~& \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N)  \\
  ///     \subjectto & \varphi(x_i, u_i, x_{i+1}) = 0, \ i \in [ 0, N-1 ] \\
  ///                & g(x_i, u_i) \in \calC_i
  ///   \end{aligned}
  /// \f]
  template<typename _Scalar>
  struct ShootingProblemTpl
  {
    using Scalar = _Scalar;
    using StageModel = StageModelTpl<Scalar>;
    using StageData = StageDataTpl<Scalar>;

    using ProblemData = ShootingProblemDataTpl<Scalar>;

    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    /// Initial condition
    VectorXs x0_init;

    /// Stages of the control problem.
    std::vector<StageModel> stages_;
    shared_ptr<CostBaseTpl<Scalar>> term_cost_;

    ShootingProblemTpl(const VectorXs& x0, const std::vector<StageModel>& stages) : x0_init(x0), stages_(stages) {}
    ShootingProblemTpl(const VectorXs& x0) : ShootingProblemTpl(x0, {}) {}

    /// @brief Add a stage to the control problem.
    void addStage(const StageModel& new_stage);
    /// @copybrief addStage()
    void addStage(StageModel&& new_stage);

    inline std::size_t numSteps() const;

    /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
    void evaluate(const std::vector<VectorXs>& xs,
                  const std::vector<VectorXs>& us,
                  ProblemData& prob_data) const;

    /// @brief Rollout the problem derivatives, stage per stage.
    ///
    /// @param xs State sequence
    /// @param us Control sequence
    void computeDerivatives(const std::vector<VectorXs>& xs,
                            const std::vector<VectorXs>& us,
                            ProblemData& prob_data) const;

    shared_ptr<ProblemData> createData() const
    {
      return std::make_shared<ProblemData>(*this);
    }

  };

  /// @brief Problem data struct.
  template<typename _Scalar>
  struct ShootingProblemDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using StageDataPtr = shared_ptr<StageDataTpl<Scalar>>;
    /// Data structs for each stage of the problem.
    std::vector<StageDataPtr> stage_data;
    /// Terminal cost data.
    shared_ptr<CostDataTpl<Scalar>> term_cost_data;

    ShootingProblemDataTpl(const ShootingProblemTpl<Scalar>& problem)
    {
      stage_data.reserve(problem.numSteps());
      for (std::size_t i = 0; i < problem.numSteps(); i++)
      {
        stage_data.push_back(std::move(problem.stages_[i].createData()));
      }

      if (problem.term_cost_)
      {
        term_cost_data = problem.term_cost_->createData();
      }
    }
  };
  
} // namespace proxddp

#include "proxddp/core/problem.hxx"

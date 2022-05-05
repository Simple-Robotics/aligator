#pragma once

#include "proxddp/core/problem.hpp"
#include "proxddp/core/solver-workspace.hpp"

#include <fmt/color.h>
#include <fmt/ostream.h>


namespace proxddp
{
  /// Storage for Riccati backward pass.
  template<typename _Scalar>
  struct WorkspaceTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    using value_storage_t = internal::value_storage<Scalar>;
    using q_storage_t = internal::q_function_storage<Scalar>;

    /// @brief Value function parameter storage
    std::vector<value_storage_t> value_params;

    /// @brief Q-function storage
    std::vector<q_storage_t> q_params;

    /// @name Riccati gains and buffers for primal-dual steps

    std::vector<MatrixXs> gains_;
    std::vector<VectorXs> dxs_;
    std::vector<VectorXs> dus_;
    std::vector<VectorXs> dlams_;

    /// Buffer for KKT matrix
    MatrixXs kktMatrixFull_;

  };

  template<typename Scalar>
  WorkspaceTpl<Scalar> createWorkspace(
    const ShootingProblemTpl<Scalar>& problem_)
  {
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    using Workspace = WorkspaceTpl<Scalar>;
    using value_storage_t = typename Workspace::value_storage_t;
    using q_storage_t = typename Workspace::q_storage_t;
    using StageModel = StageModelTpl<Scalar>;

    Workspace workspace;
    const std::size_t nsteps = problem_.numSteps();
    workspace.value_params.reserve(nsteps);

    int nprim;
    int ndual;
    for (std::size_t i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem_.stages_[i];
      nprim = stage.numPrimal();
      ndual = stage.numDual();
      workspace.value_params.push_back(value_storage_t(stage.ndx1()));
      workspace.q_params.push_back(q_function_storage(stage.ndx1(), stage.nu(), stage.ndx2()));

      workspace.gains_.push_back(MatrixXs::Zero(nprim + ndual, stage.ndx1() + 1));

      workspace.dxs_.push_back(VectorXs::Zero(stage.ndx1()));
      workspace.dus_.push_back(VectorXs::Zero(stage.nu()));
      workspace.dlams_.push_back(VectorXs::Zero(ndual));

    }
    // terminal node
    const int term_ndx = problem_.stages_[nsteps - 1].ndx2();
    workspace.value_params.push_back(value_storage_t(term_ndx));
    workspace.dxs_.push_back(VectorXs::Zero(term_ndx));

    assert(workspace.value_params.size() == nsteps + 1);
    assert(workspace.dxs_.size() == nsteps + 1);
    assert(workspace.dus_.size() == nsteps);

    return workspace;
  }


  /// Run the Riccati forward pass.
  template<typename Scalar>
  void forward_pass(
    const ShootingProblemTpl<Scalar>& problem,
    WorkspaceTpl<Scalar>& workspace)
  {
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
    const std::vector<MatrixXs>& gains = workspace.gains_;

  }

  
} // namespace proxddp



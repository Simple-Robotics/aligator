#pragma once
#include "proxddp/fwd.hpp"


namespace proxddp
{
  namespace internal
  {
    
    /// @brief  Contiguous storage for the value function parameters.
    ///
    /// @details This provides storage for the matrix \f[
    ///     \begin{bmatrix} 2v & V_x^\top \\ V_x & V_{xx} \end{bmatrix}
    /// \f]
    template<typename Scalar>
    struct value_storage
    {
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      MatrixXs storage;
      Scalar& v_2_;
      VectorRef Vx_;
      MatrixRef Vxx_;

      value_storage(const int ndx)
        : storage(ndx + 1, ndx + 1)
        , v_2_(storage.coeffRef(0, 0))
        , Vx_(storage.bottomLeftCorner(ndx, 1))
        , Vxx_(storage.bottomRightCorner(ndx, ndx)) {
        storage.setZero();
      }
    };

    template<typename Scalar>
    struct q_function_storage
    {
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      const int ntot;

      MatrixXs storage;

      Scalar& q_2_; 

      VectorRef grad_;
      MatrixRef hess_;

      VectorRef Qx_;
      VectorRef Qu_;
      VectorRef Qy_;

      MatrixRef Qxx_;
      MatrixRef Qxu_;
      MatrixRef Qxy_;
      MatrixRef Quu_;
      MatrixRef Quy_;
      MatrixRef Qyy_;

      q_function_storage(const int ndx1, const int nu, const int ndx2)
        : ntot(ndx1 + nu + ndx2)
        , storage(ntot + 1, ntot + 1)
        , q_2_(storage.coeffRef(0, 0))
        , grad_(storage.bottomLeftCorner(ntot, 1))
        , hess_(storage.bottomRightCorner(ntot, ntot))
        , Qx_(grad_.head(ndx1))
        , Qu_(grad_.segment(ndx1, nu))
        , Qy_(grad_.tail(ndx2))
        , Qxx_(hess_.topLeftCorner(ndx1, ndx1))
        , Qxu_(hess_.block(0, ndx1, ndx1, nu))
        , Qxy_(hess_.topRightCorner(ndx1, ndx2))
        , Quu_(hess_.block(ndx1, ndx1, nu, nu))
        , Quy_(hess_.block(ndx1, ndx1 + nu, nu, ndx2))
        , Qyy_(hess_.bottomRightCorner(ndx2, ndx2)) {
        storage.setZero();
      }

    };

  } // namespace internal


  /// @brief Storage for the Riccati forward and backward passes.
  template<typename _Scalar>
  struct WorkspaceTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
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

    /// @name Temp data

    std::vector<VectorXs> trial_xs_;
    std::vector<VectorXs> trial_us_;
    std::vector<VectorXs> trial_lams_;

    WorkspaceTpl(const ShootingProblemTpl<Scalar>& problem);

  };


  template<typename Scalar>
  WorkspaceTpl<Scalar>::WorkspaceTpl(const ShootingProblemTpl<Scalar>& problem)
  {
    using VectorXs = typename math_types<Scalar>::VectorXs;
    using MatrixXs = typename math_types<Scalar>::MatrixXs;
    using Workspace = WorkspaceTpl<Scalar>;
    using value_storage_t = typename Workspace::value_storage_t;
    using q_storage_t = typename Workspace::q_storage_t;
    using StageModel = StageModelTpl<Scalar>;

    const std::size_t nsteps = problem.numSteps();
    value_params.reserve(nsteps);

    trial_xs_.reserve(nsteps + 1);
    trial_us_.reserve(nsteps);
    trial_lams_.reserve(nsteps);

    int nprim;
    int ndual;
    int nx;
    int nu;
    int max_kkt_size = 0;
    std::size_t i = 0;
    for (i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      nx = stage.xspace1_.nx();
      nu = stage.nu();
      nprim = stage.numPrimal();
      ndual = stage.numDual();

      value_params.push_back(value_storage_t(stage.ndx1()));
      q_params.push_back(q_storage_t(stage.ndx1(), nu, stage.ndx2()));

      gains_.push_back(MatrixXs::Zero(nprim + ndual, stage.ndx1() + 1));

      dxs_.push_back(VectorXs::Zero(stage.ndx1()));
      dus_.push_back(VectorXs::Zero(nu));
      dlams_.push_back(VectorXs::Zero(ndual));

      trial_xs_.push_back(VectorXs::Zero(nx));
      trial_us_.push_back(VectorXs::Zero(nu));
      trial_lams_.push_back(VectorXs::Zero(ndual));

      max_kkt_size = std::max(max_kkt_size, nprim + ndual);
    }

    kktMatrixFull_.resize(max_kkt_size, max_kkt_size);
    kktMatrixFull_.setZero();

    assert(i == nsteps);
    // terminal node
    const auto& stage = problem.stages_[nsteps - 1];

    nx = stage.xspace1_.nx();
    value_params.push_back(value_storage_t(stage.ndx2()));

    dxs_.push_back(VectorXs::Zero(stage.ndx2()));
    trial_xs_.push_back(VectorXs::Zero(nx));

    assert(value_params.size() == nsteps + 1);
    assert(dxs_.size() == nsteps + 1);
    assert(dus_.size() == nsteps);
  }

} // namespace proxddp

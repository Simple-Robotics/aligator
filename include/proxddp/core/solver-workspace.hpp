#pragma once
#include "proxddp/fwd.hpp"

#include <ostream>


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
      // Eigen::SelfAdjointView<MatrixXs, Eigen::Lower> store_sym_;
      Scalar& v_2_;
      VectorRef Vx_;
      MatrixRef Vxx_;

      value_storage(const int ndx)
        : storage(MatrixXs::Zero(ndx + 1, ndx + 1))
        // , store_sym_(storage)
        , v_2_(storage.coeffRef(0, 0))
        , Vx_(storage.bottomLeftCorner(ndx, 1))
        , Vxx_(storage.bottomRightCorner(ndx, ndx))
        {}

      friend std::ostream& operator<<(std::ostream& oss, const value_storage& store)
      {
        Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
        oss << "value_storage {\n";
        oss << store.storage.format(CleanFmt);
        // oss << MatrixXs(store.store_sym_).format(CleanFmt);
        oss << "\n}";
        return oss;
      }
    };

    /** @brief  Contiguous storage for Q-function parameters with corresponding
     *          sub-matrix views.
     * 
     * @details  The storage layout is as follows:
     * \f[
     *    \begin{bmatrix}
     *      2q    & Q_x^\top  & Q_u^top & Q_y^\top  \\
     *      Q_x   & Q_{xx}    & Q_{xu}  & Q_{xy}    \\
     *      Q_u   & Q_{ux}    & Q_{uu}  & Q_{uy}    \\
     *      Q_y   & Q_{yx}    & Q_{yu}  & Q_{yy} 
     *    \end{bmatrix}
     * ]\f
     */
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


  /// @brief    Storage for the Riccati forward and backward passes.
  template<typename _Scalar>
  struct WorkspaceTpl
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using value_storage_t = internal::value_storage<Scalar>;
    using q_storage_t = internal::q_function_storage<Scalar>;

    const std::size_t nsteps;

    shared_ptr<ShootingProblemDataTpl<Scalar>> problem_data;

    /// @brief Value function parameter storage
    std::vector<value_storage_t> value_params;

    /// @brief Q-function storage
    std::vector<q_storage_t> q_params;

    std::vector<VectorXs> lams_plus_;
    std::vector<VectorXs> lams_pdal_;

    /// @name Riccati gains and buffers for primal-dual steps

    std::vector<MatrixXs> gains_;
    std::vector<VectorXs> pd_step_;
    std::vector<VectorRef> dxs_;
    std::vector<VectorRef> dus_;
    std::vector<VectorRef> dlams_;

    /// Buffer for KKT matrix
    MatrixXs kktMatrixFull_;
    /// Buffer for KKT right hand side
    MatrixXs kktRhsFull_;

    /// @name Temp data

    std::vector<VectorXs> trial_xs_;
    std::vector<VectorXs> trial_us_;
    std::vector<VectorXs> trial_lams_;

    /// @name Previous proximal iterates

    std::vector<VectorXs> prev_xs_;
    std::vector<VectorXs> prev_us_;
    std::vector<VectorXs> prev_lams_;

    Scalar inner_criterion;
    VectorXs inner_criterion_by_stage;
    Scalar primal_infeasibility;
    Scalar dual_infeasibility;

    explicit WorkspaceTpl(const ShootingProblemTpl<Scalar>& problem);

    MatrixRef
    getKktView(const int nprim, const int ndual)
    {
      return kktMatrixFull_
        .topLeftCorner(nprim + ndual, nprim + ndual);
    }

    MatrixRef
    getKktRhs(const int nprim, const int ndual, const int ndx1)
    {
      return kktRhsFull_.topLeftCorner(nprim + ndual, ndx1 + 1);
    }

    friend std::ostream& operator<<(std::ostream& oss, const WorkspaceTpl<Scalar>& workspace)
    {
      oss << "Workspace {";
      oss << "\n\tnum nodes:              " << workspace.trial_us_.size()
          << "\n\tkkt matrix buffer size: " << workspace.kktMatrixFull_.rows();
      oss << "\n}";
      return oss;
    }
  };

  template<typename Scalar>
  WorkspaceTpl<Scalar>::WorkspaceTpl(const ShootingProblemTpl<Scalar>& problem)
    : nsteps(problem.numSteps())
    , problem_data(problem.createData())
    , inner_criterion_by_stage(nsteps)
  {
    using VectorXs = typename math_types<Scalar>::VectorXs;
    using MatrixXs = typename math_types<Scalar>::MatrixXs;
    using Workspace = WorkspaceTpl<Scalar>;
    using value_storage_t = typename Workspace::value_storage_t;
    using q_storage_t = typename Workspace::q_storage_t;
    using StageModel = StageModelTpl<Scalar>;

    inner_criterion_by_stage.setZero();

    value_params.reserve(nsteps + 1);
    q_params.reserve(nsteps);

    lams_plus_.reserve(nsteps);
    lams_pdal_.reserve(nsteps);

    trial_xs_.reserve(nsteps + 1);
    trial_us_.reserve(nsteps);
    trial_lams_.reserve(nsteps);
    prev_xs_.reserve(nsteps + 1);
    prev_us_.reserve(nsteps);
    prev_lams_.reserve(nsteps);

    int nprim, ndual;
    int ndx1, nu, ndx2;
    int max_kkt_size = 0;
    int max_ndx = problem.stages_[0].ndx1();
    ndx1 = problem.stages_[0].ndx1();
    pd_step_.push_back(VectorXs::Zero(ndx1));
    dxs_.push_back(pd_step_[0].head(ndx1));

    std::size_t i = 0;
    for (i = 0; i < nsteps; i++)
    {
      const StageModel& stage = problem.stages_[i];
      ndx1 = stage.ndx1(),
      nu = stage.nu();
      ndx2 = stage.ndx2();
      nprim = stage.numPrimal();
      ndual = stage.numDual();

      value_params.push_back(value_storage_t(ndx1));
      q_params.push_back(q_storage_t(ndx1, nu, ndx2));

      lams_plus_.push_back(VectorXs::Zero(ndual));
      lams_pdal_.push_back(VectorXs::Zero(ndual));

      gains_.push_back(MatrixXs::Zero(nprim + ndual, ndx1 + 1));

      pd_step_.push_back(VectorXs::Zero(nprim + ndual));
      // dxs_.push_back(VectorXs::Zero(ndx1));
      // dus_.push_back(VectorXs::Zero(nu));
      // dlams_.push_back(VectorXs::Zero(ndual));
      dxs_.push_back(pd_step_[i + 1].segment(nu, ndx2));
      dus_.push_back(pd_step_[i + 1].head(nu));
      dlams_.push_back(pd_step_[i + 1].tail(ndual));

      trial_xs_.push_back(VectorXs::Zero(stage.nx1()));
      trial_us_.push_back(VectorXs::Zero(nu));
      trial_lams_.push_back(VectorXs::Zero(ndual));

      prev_xs_.push_back(trial_xs_[i]);
      prev_us_.push_back(trial_us_[i]);
      prev_lams_.push_back(trial_lams_[i]);

      /** terminal node **/
      if (i == nsteps - 1)
      {
        value_params.push_back(value_storage_t(ndx2));
        trial_xs_.push_back(VectorXs::Zero(stage.nx2()));
        prev_xs_.push_back(trial_xs_[nsteps]);
      }

      max_kkt_size = std::max(max_kkt_size, nprim + ndual);
      max_ndx = std::max(max_ndx, ndx2);
    }

    kktMatrixFull_.resize(max_kkt_size, max_kkt_size);
    kktMatrixFull_.setZero();

    kktRhsFull_.resize(max_kkt_size, max_ndx + 1);;
    kktRhsFull_.setZero();

    assert(value_params.size() == nsteps + 1);
    assert(dxs_.size() == nsteps + 1);
    assert(dus_.size() == nsteps);
  }

} // namespace proxddp

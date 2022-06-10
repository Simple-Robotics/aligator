

namespace proxddp
{

  template<typename Scalar>
  WorkspaceTpl<Scalar>::WorkspaceTpl(const ShootingProblemTpl<Scalar>& problem)
    : nsteps(problem.numSteps())
    , problem_data(problem.createData())
    , inner_criterion_by_stage(nsteps)
    , primal_infeas_by_stage(nsteps)
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


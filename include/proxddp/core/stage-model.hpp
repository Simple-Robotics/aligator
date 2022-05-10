#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>

#include "proxddp/core/costs.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/constraint.hpp"

#include "proxddp/core/clone.hpp"


namespace proxddp
{
  // fwd StageData
  template<typename _Scalar>
  struct StageDataTpl;

  template<typename Scalar>
  struct ConstraintContainer
  {
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Constraint = StageConstraintTpl<Scalar>;
    ConstraintContainer() {};

    std::size_t size() const
    {
      return storage_.size();
    }

    void push_back(const shared_ptr<Constraint>& el)
    {
      const int nr = el->nr();
      const int last_cursor = cursors_[this->size() - 1];
      storage_.push_back(el);
      cursors_.push_back(last_cursor + nr);
      dims_.push_back(nr);
    }

    std::size_t getIndex(const std::size_t i) const
    {
      return cursors_[i];
    }

    std::size_t getDim(const std::size_t i) const
    {
      return dims_[i];
    }

    /// Get corresponding segment of a vector corresponding
    /// to the @p i-th constraint.
    VectorRef getSegmentByConstraint(VectorRef lambda, const std::size_t i) const
    {
      assert(lambda.size() == totalDim());
      return lambda.segment(getIndex(i), getDim(i));
    }

    MatrixRef getBlockByConstraint(MatrixRef J, const std::size_t i) const
    {
      assert(J.rows() == totalDim());
      return J.middleRows(getIndex(i), getDim(i));
    }

    std::size_t totalDim() const
    {
      return cursors_[this->size() - 1];
    }

    shared_ptr<Constraint>& operator[](std::size_t i)
    {
      return storage_[i];
    }

    const shared_ptr<Constraint>& operator[](std::size_t i) const
    {
      return storage_[i];
    }

  protected:
    std::vector<shared_ptr<Constraint>> storage_;
    std::vector<int> cursors_;
    std::vector<int> dims_;
    void update_indices()
    {
      cursors_.clear();
      int cursor = 0;
      int nr = 0;
      for (std::size_t i = 0; i < storage_.size(); i++)
      {
        nr = storage_[i]->nr();
        cursors_.push_back(cursor);
        dims_.push_back(nr);
        cursor += nr;
      }
    }
  };

  /** @brief    A stage in the control problem.
   * 
   *  @details  Each stage containts cost functions, dynamical
   *            and constraint models.
   */
  template<typename _Scalar>
  class StageModelTpl : public cloneable<StageModelTpl<_Scalar>>
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using Manifold = ManifoldAbstractTpl<Scalar>;
    using Dynamics = DynamicsModelTpl<Scalar>;
    using Constraint = StageConstraintTpl<Scalar>;
    using CostBase = CostBaseTpl<Scalar>;
    using Data = StageDataTpl<Scalar>;

    using ConstraintPtr = shared_ptr<Constraint>;

    /// Current state space.
    const Manifold& xspace1_;
    /// Next state space.
    const Manifold& xspace2_;
    /// Control vector space.
    proxnlp::VectorSpaceTpl<Scalar> uspace;

    const CostBase& cost_;
    const Dynamics& dyn_model_;
    ConstraintContainer<Scalar> constraints_;

    inline int nx1()  const { return xspace1_.nx(); }
    inline int ndx1() const { return xspace1_.ndx(); }
    inline int nu()   const { return uspace.ndx(); }
    inline int nx2()  const { return xspace2_.nx(); }
    inline int ndx2() const { return xspace2_.ndx(); }

    inline std::size_t numConstraints() const { return constraints_.size(); }

    /// Number of primal optimization variables.
    int numPrimal() const;
    /// Number of dual variables, i.e. Lagrange multipliers.
    int numDual() const;

    StageModelTpl(const Manifold& space1,
                  const int nu,
                  const Manifold& space2,
                  const CostBase& cost,
                  const Dynamics& dyn_model)
      : xspace1_(space1)
      , xspace2_(space2)
      , uspace(nu)
      , cost_(cost)
      , dyn_model_(dyn_model)
      {}

    /// Secondary constructor: use a single manifold.
    StageModelTpl(const Manifold& space,
                  const int nu,
                  const CostBase& cost,
                  const Dynamics& dyn_model)
      : StageModelTpl(space, nu, space, cost, dyn_model)
      {}

    /// @brief    Add a constraint to the stage.
    void addConstraint(const ConstraintPtr& cstr) { constraints_.push_back(cstr); }

    /* Compute on the node */

    /// @brief    Evaluate all the functions (cost, dynamics, constraints) at this node.
    void evaluate(const ConstVectorRef& x,
                  const ConstVectorRef& u,
                  const ConstVectorRef& y,
                  Data& data) const
    {
      dyn_model_.evaluate(x, u, y, *data.dyn_data);
      cost_.evaluate(x, u, *data.cost_data);

      for (std::size_t i = 0; i < numConstraints(); i++)
      {
        // calc on constraint
        const auto& cstr = constraints_[i];
        cstr->func_.evaluate(x, u, y, *data.constraint_data[i]);
      }
    }

    /// @brief    Compute the derivatives of the StageModelTpl.
    void computeDerivatives(const ConstVectorRef& x,
                            const ConstVectorRef& u,
                            const ConstVectorRef& y,
                            Data& data) const
    {
      cost_.computeGradients(x, u, *data.cost_data);
      cost_.computeHessians(x, u, *data.cost_data);

      dyn_model_.computeJacobians(x, u, y, *data.dyn_data);

      for (std::size_t i = 0; i < numConstraints(); i++)
      {
        // calc on constraint
        const auto& cstr = constraints_[i];
        cstr->func_.computeJacobians(x, u, y, *data.constraint_data[i]);
      }
    }

    /// @brief    Create a Data object.
    shared_ptr<Data> createData() const
    {
      return std::make_shared<Data>(*this);
    }

    friend std::ostream& operator<<(std::ostream& oss, const StageModelTpl& stage)
    {
      oss << "StageModel { ";
      if (stage.ndx1() == stage.ndx2())
      {
        oss << "ndx: " << stage.ndx1() << ", "
            << "nu:  " << stage.nu();
      } else {
        oss << "ndx1:" << stage.ndx1() << ", "
            << "nu:  " << stage.nu() << ", "
            << "ndx2:" << stage.ndx2();
      }

      if (stage.numConstraints() > 0)
      {
        oss << ", ";
        oss << "nc: " << stage.numConstraints();
      }
      
      oss << " }";
      return oss;
    }
  };

  /// @brief    Data struct for stage models StageModelTpl.
  template<typename _Scalar>
  struct StageDataTpl : public cloneable<StageDataTpl<_Scalar>>
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar);

    using StageModel = StageModelTpl<Scalar>;
    using CostData = CostDataTpl<Scalar>;
    using FunctionData = FunctionDataTpl<Scalar>;

    /// Data struct for the dynamics.
    shared_ptr<DynamicsDataTpl<Scalar>> dyn_data;
    /// Data structs for the functions involved in the constraints.
    std::vector<shared_ptr<FunctionData>> constraint_data;
    /// Data for the running costs.
    shared_ptr<CostData> cost_data;

    /// @brief    Constructor.
    ///
    /// @details  The constructor initializes or fills in the data members using move semantics.
    explicit StageDataTpl(const StageModel& stage_model)
      : dyn_data(std::move(stage_model.dyn_model_.createData()))
      , cost_data(std::move(stage_model.cost_.createData()))
    {
      const std::size_t nc = stage_model.numConstraints();
      constraint_data.reserve(nc);
      for (std::size_t i = 0; i < nc; i++)
      {
        const auto& func = stage_model.constraints_[i]->func_;
        constraint_data.push_back(std::move(func.createData()));
      }
    }

  };

} // namespace proxddp

#include "proxddp/core/stage-model.hxx"

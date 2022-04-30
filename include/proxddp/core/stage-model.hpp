#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/node-function.hpp"

#include <proxnlp/manifold-base.hpp>

#include <proxnlp/modelling/spaces/cartesian-product.hpp>
#include <proxnlp/modelling/spaces/vector-space.hpp>

#include "proxddp/core/costs.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/constraint.hpp"


namespace proxddp
{
  using proxnlp::VectorSpaceTpl;
  using proxnlp::CartesianProductTpl;


  // fwd StageData
  template<typename _Scalar>
  struct StageDataTpl;


  /** @brief    A stage in the control problem.
   * 
   *  @details  Each stage containts cost functions, dynamical
   *            and constraint models.
   */
  template<typename _Scalar>
  class StageModelTpl
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)

    using Manifold = ManifoldAbstractTpl<Scalar>;
    using Dynamics = DynamicsModelTpl<Scalar>;
    using Constraint = StageConstraintTpl<Scalar>;
    using Data = StageDataTpl<Scalar>;

    using ConstraintPtr = shared_ptr<Constraint>;

    /// Current state space.
    const Manifold& xspace1_;
    /// Next state space.
    const Manifold& xspace2_;
    /// Control vector space.
    VectorSpaceTpl<Scalar> uspace;

    const Dynamics& dyn_model_;
    std::vector<ConstraintPtr> constraints_;

    inline int ndx1() const { return xspace1_.ndx(); }
    inline int nu()   const { return uspace.ndx(); }
    inline int ndx2() const { return xspace2_.ndx(); }

    inline std::size_t numConstraints() const { return constraints_.size(); }

    StageModelTpl(const Manifold& space1,
                  const int nu,
                  const Manifold& space2,
                  const Dynamics& dyn_model)
      : xspace1_(space1)
      , xspace2_(space2)
      , uspace(nu)
      , dyn_model_(dyn_model)
      {}

    /// Secondary constructor: use a single manifold.
    StageModelTpl(const Manifold& space, const int nu, const Dynamics& dyn_model)
      : StageModelTpl(space, nu, space, dyn_model)
      {}


    /// @brief    Add a constraint to the stage.
    void addConstraint(const ConstraintPtr& cstr) { constraints_.push_back(cstr); }
    /// @copybrief addConstraint()
    /// @details   This moves the rvalue into the vector.
    void addConstraint(ConstraintPtr&& cstr) { constraints_.push_back(std::move(cstr)); }

    /* Compute on the node */

    /// @brief    Evaluate all the functions (cost, dynamics, constraints) at this node.
    void calc(const ConstVectorRef& x,
              const ConstVectorRef& u,
              const ConstVectorRef& y,
              Data& data) const
    {
      dyn_model_->evaluate(x, u, y, *data.dyn_data);
      for (std::size_t i = 0; i < numConstraints(); i++)
      {
        // calc on constraint
        auto& cstr = constraints_[i];
        cstr->func_.evaluate(x, u, y, *data.constraint_data[i]);
      }
    }


    /// @brief    Create a Data object.
    shared_ptr<Data> createData() const
    {
      return std::make_shared<Data>(*this);
    }

    friend std::ostream& operator<<(std::ostream& oss, const StageModelTpl& stage)
    {
      oss << "StageModel(";
      if (stage.ndx1() == stage.ndx2())
      {
        oss << "ndx=" << stage.ndx1()
            << ", nu=" << stage.nu();
      } else {
        oss << "ndx1=" << stage.ndx1() << ", "
            << "nu=" << stage.nu() << ", "
            << "ndx2=" << stage.ndx2();
      }

      if (stage.numConstraints() > 0)
      {
        oss << ", ";
        oss << "nc=" << stage.numConstraints();
      }
      
      oss << ")";
      return oss;
    }
  };

  /// @brief    Data struct for stage models StageModelTpl.
  template<typename _Scalar>
  struct StageDataTpl
  {
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)

    using StageModel = StageModelTpl<Scalar>;
    using FunctionData = FunctionDataTpl<Scalar>;

    /// Data struct for the dynamics.
    shared_ptr<DynamicsDataTpl<Scalar>> dyn_data;
    /// Data structs for the functions involved in the constraints.
    std::vector<shared_ptr<FunctionData>> constraint_data;

    /// @brief    Constructor.
    ///
    /// The constructor initializes or fills in the data members using move semantics.
    explicit StageDataTpl(const StageModel& stage_model)
      : dyn_data(std::move(stage_model.dyn_model_.createData()))
    {
      const std::size_t nc = stage_model.numConstraints();
      constraint_data.reserve(nc);
      for (std::size_t i = 0; i < nc; i++)
      {
        const auto& func = stage_model.constraints_[i]->func_;
        constraint_data.emplace_back(std::move(func.createData()));
      }
    }

  };
  


} // namespace proxddp

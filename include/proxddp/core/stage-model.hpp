/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/cost-abstract.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/constraint.hpp"
#include "proxddp/utils/exceptions.hpp"

#include "proxddp/core/clone.hpp"

namespace proxddp {

/** @brief    A stage in the control problem.
 *
 *  @details  Each stage containts cost functions, dynamical
 *            and constraint models. These objects are hold
 *            through smart pointers to leverage dynamic
 *            polymorphism.
 */
template <typename _Scalar>
struct StageModelTpl : Cloneable<StageModelTpl<_Scalar>> {
public:
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);

  using Manifold = ManifoldAbstractTpl<Scalar>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using Dynamics = DynamicsModelTpl<Scalar>;
  using DynamicsPtr = shared_ptr<Dynamics>;
  using FunctionPtr = shared_ptr<StageFunctionTpl<Scalar>>;
  using ConstraintSetPtr = shared_ptr<ConstraintSetBase<Scalar>>;
  using Constraint = StageConstraintTpl<Scalar>;
  using Cost = CostAbstractTpl<Scalar>;
  using CostPtr = shared_ptr<Cost>;
  using Data = StageDataTpl<Scalar>;

  /// State space for the current state \f$x_k\f$.
  ManifoldPtr xspace_;
  /// State space for the next state \f$x_{k+1}\f$.
  ManifoldPtr xspace_next_;
  /// Control vector space -- by default, a simple Euclidean space.
  ManifoldPtr uspace_;
  /// Stage cost function.
  CostPtr cost_;
  /// Constraint manager.
  ConstraintStackTpl<Scalar> constraints_;

  /// Constructor assumes the control space is a Euclidean space of
  /// dimension @p nu.
  StageModelTpl(CostPtr cost, DynamicsPtr dyn_model);
  virtual ~StageModelTpl() = default;

  const Manifold &xspace() const { return *xspace_; }
  const Manifold &uspace() const { return *uspace_; }
  const Manifold &xspace_next() const { return *xspace_next_; }

  const Cost &cost() const { return *cost_; }
  /// Whether the stage's dynamics model can be accessed.
  /// This boolean allows flexibility in solvers when dealing
  /// with different frontends e.g. Crocoddyl's API.
  virtual bool has_dyn_model() const { return true; }
  virtual const Dynamics &dyn_model() const {
    assert(numConstraints() > 0);
    auto dyn_ptr = std::static_pointer_cast<Dynamics>(constraints_[0].func);
    return *dyn_ptr;
  }

  int nx1() const { return xspace_->nx(); }
  int ndx1() const { return xspace_->ndx(); }
  int nu() const { return uspace_->ndx(); }
  int nx2() const { return xspace_next_->nx(); }
  int ndx2() const { return xspace_next_->ndx(); }

  /// Number of constraints (constraint objects).
  std::size_t numConstraints() const { return constraints_.size(); }

  /// Number of primal optimization variables.
  int numPrimal() const;
  /// Number of dual variables, i.e. Lagrange multipliers.
  int numDual() const;

  /// @brief    Add a constraint to the stage.
  template <typename T> void addConstraint(T &&cstr);

  /// @copybrief  addConstraint().
  /// @details    Adds a constraint by allocating a new StageConstraintTpl.
  void addConstraint(FunctionPtr func, ConstraintSetPtr cstr_set);

  /* Evaluate costs, constraints, ... */

  /// @brief    Evaluate all the functions (cost, dynamics, constraints) at this
  /// node.
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const;

  /// @brief    Compute the derivatives of the StageModelTpl.
  virtual void computeDerivatives(const ConstVectorRef &x,
                                  const ConstVectorRef &u,
                                  const ConstVectorRef &y, Data &data) const;

  /// @brief    Create a Data object.
  virtual shared_ptr<Data> createData() const;

  friend std::ostream &operator<<(std::ostream &oss,
                                  const StageModelTpl &stage) {
    oss << "StageModel { ";
    if (stage.ndx1() == stage.ndx2()) {
      oss << "ndx: " << stage.ndx1() << ", "
          << "nu:  " << stage.nu();
    } else {
      oss << "ndx1:" << stage.ndx1() << ", "
          << "nu:  " << stage.nu() << ", "
          << "ndx2:" << stage.ndx2();
    }

    if (stage.numConstraints() > 0) {
      oss << ", ";
      oss << "nc: " << stage.numConstraints();
    }

    oss << " }";
    return oss;
  }

protected:
  StageModelTpl(ManifoldPtr space, const int nu);
  virtual StageModelTpl *clone_impl() const override {
    return new StageModelTpl(*this);
  }
};

} // namespace proxddp

template <typename Scalar>
struct fmt::formatter<proxddp::StageModelTpl<Scalar>> : fmt::ostream_formatter {
};

#include "proxddp/core/stage-model.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/stage-model.txx"
#endif

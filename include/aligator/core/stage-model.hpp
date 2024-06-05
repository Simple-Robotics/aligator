/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/dynamics.hpp"
#include "aligator/core/constraint.hpp"
#include "aligator/core/common-model-container.hpp"
#include "aligator/core/common-model-builder-container.hpp"
#include "aligator/utils/exceptions.hpp"

#include "aligator/core/clone.hpp"

#include <boost/optional.hpp>

namespace aligator {

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
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

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
  using CommonModel = CommonModelTpl<Scalar>;
  using CommonModelContainer = CommonModelContainerTpl<Scalar>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<Scalar>;

  /// State space for the current state \f$x_k\f$.
  ManifoldPtr xspace_;
  /// State space for the next state \f$x_{k+1}\f$.
  ManifoldPtr xspace_next_;
  /// Control vector space -- by default, a simple Euclidean space.
  ManifoldPtr uspace_;
  /// Constraint manager.
  ConstraintStackTpl<Scalar> constraints_;
  /// Stage cost function.
  CostPtr cost_;
  /// Dynamics model
  DynamicsPtr dynamics_;
  // TODO to avoid mutable, we must call configure outside setup
  // TODO or remove setup constness
  /// Store all CommonModel
  mutable CommonModelContainer common_model_container_;
  /// Contains all CommonModelBuilder
  mutable CommonModelBuilderContainer common_model_builder_container_;

  /// Constructor assumes the control space is a Euclidean space of
  /// dimension @p nu.
  StageModelTpl(CostPtr cost, DynamicsPtr dynamics);
  virtual ~StageModelTpl() = default;

  const Manifold &xspace() const { return *xspace_; }
  const Manifold &uspace() const { return *uspace_; }
  const Manifold &xspace_next() const { return *xspace_next_; }

  const Cost &cost() const { return *cost_; }
  /// Whether the stage's dynamics model can be accessed.
  /// This boolean allows flexibility in solvers when dealing
  /// with different frontends e.g. Crocoddyl's API.
  virtual bool has_dyn_model() const { return true; }
  ALIGATOR_DEPRECATED virtual const Dynamics &dyn_model() const {
    return *dynamics_;
  }

  int nx1() const { return xspace_->nx(); }
  int ndx1() const { return xspace_->ndx(); }
  int nu() const { return uspace_->ndx(); }
  int nx2() const { return xspace_next_->nx(); }
  int ndx2() const { return xspace_next_->ndx(); }
  /// Total number of constraints
  int nc() const { return (int)constraints_.totalDim(); }

  /// Number of constraint objects.
  std::size_t numConstraints() const { return constraints_.size(); }

  /// Number of primal optimization variables.
  int numPrimal() const { return nu() + ndx2(); }
  /// Number of dual variables, i.e. Lagrange multipliers.
  int numDual() const { return ndx2() + nc(); }

  /// @brief    Add a constraint to the stage.
  template <typename T> void addConstraint(T &&cstr);

  /// @copybrief  addConstraint().
  /// @details    Adds a constraint by allocating a new StageConstraintTpl.
  void addConstraint(FunctionPtr func, ConstraintSetPtr cstr_set);

  /* Configure costs, constraints, ... */
  virtual void configure() const;

  /* Evaluate costs, constraints, ... */

  /// @brief    Evaluate all the functions (cost, dynamics, constraints) at this
  /// node.
  virtual void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const;

  /// @brief    Compute the first-order derivatives of the StageModelTpl.
  virtual void computeFirstOrderDerivatives(const ConstVectorRef &x,
                                            const ConstVectorRef &u,
                                            const ConstVectorRef &y,
                                            Data &data) const;

  /// @brief    Compute the second-order derivatives of the StageModelTpl.
  virtual void computeSecondOrderDerivatives(const ConstVectorRef &x,
                                             const ConstVectorRef &u,
                                             Data &data) const;

  /// @brief    Create a StageData object.
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

} // namespace aligator

template <typename Scalar>
struct fmt::formatter<aligator::StageModelTpl<Scalar>>
    : fmt::ostream_formatter {};

#include "aligator/core/stage-model.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/stage-model.txx"
#endif

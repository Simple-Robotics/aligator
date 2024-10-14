/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/dynamics.hpp"
#include "aligator/core/constraint.hpp"

#include <fmt/ostream.h>

namespace aligator {

#define ALIGATOR_CHECK_DERIVED_CLASS(Base, Derived)                            \
  static_assert((std::is_base_of_v<Base, Derived>),                            \
                "Failed check for derived class.")

/** @brief    A stage in the control problem.
 *
 *  @details  Each stage containts cost functions, dynamical
 *            and constraint models. These objects are hold
 *            through smart pointers to leverage dynamic
 *            polymorphism.
 */
template <typename _Scalar> struct StageModelTpl {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using Manifold = ManifoldAbstractTpl<Scalar>;
  using PolyManifold = xyz::polymorphic<Manifold>;
  using Dynamics = DynamicsModelTpl<Scalar>;
  using PolyDynamics = xyz::polymorphic<Dynamics>;
  using PolyFunction = xyz::polymorphic<StageFunctionTpl<Scalar>>;
  using PolyConstraintSet = xyz::polymorphic<ConstraintSetTpl<Scalar>>;
  using Cost = CostAbstractTpl<Scalar>;
  using PolyCost = xyz::polymorphic<Cost>;
  using Data = StageDataTpl<Scalar>;

  /// State space for the current state \f$x_k\f$.
  PolyManifold xspace_;
  /// State space for the next state \f$x_{k+1}\f$.
  PolyManifold xspace_next_;
  /// Control vector space -- by default, a simple Euclidean space.
  PolyManifold uspace_;
  /// Constraint manager.
  ConstraintStackTpl<Scalar> constraints_;
  /// Stage cost function.
  PolyCost cost_;
  /// Dynamics model
  PolyDynamics dynamics_;

  /// @brief Get a pointer to an expected concrete type for the cost function.
  template <typename U> U *getCost() {
    ALIGATOR_CHECK_DERIVED_CLASS(Cost, U);
    return dynamic_cast<U *>(&*cost_);
  }

  /// @copybrief castCost()
  template <typename U> const U *getCost() const {
    ALIGATOR_CHECK_DERIVED_CLASS(Cost, U);
    return dynamic_cast<const U *>(&*cost_);
  }

  /// @brief Get a pointer to an expected concrete type for the dynamics class.
  template <typename U> U *getDynamics() {
    ALIGATOR_CHECK_DERIVED_CLASS(Dynamics, U);
    return dynamic_cast<U *>(&*dynamics_);
  }

  /// @copybrief castDynamics()
  template <typename U> const U *getDynamics() const {
    ALIGATOR_CHECK_DERIVED_CLASS(Dynamics, U);
    return dynamic_cast<const U *>(&*dynamics_);
  }

  /// Constructor assumes the control space is a Euclidean space of
  /// dimension @p nu.
  StageModelTpl(const PolyCost &cost, const PolyDynamics &dynamics);
  virtual ~StageModelTpl() = default;

  const Manifold &xspace() const { return *xspace_; }
  const Manifold &uspace() const { return *uspace_; }
  const Manifold &xspace_next() const { return *xspace_next_; }

  const Cost &cost() const { return *cost_; }
  /// Whether the stage's dynamics model can be accessed.
  /// This boolean allows flexibility in solvers when dealing
  /// with different frontends e.g. Crocoddyl's API.
  virtual bool hasDynModel() const { return true; }

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
  template <typename Cstr, typename = std::enable_if_t<std::is_same_v<
                               std::decay_t<Cstr>, StageConstraintTpl<Scalar>>>>
  ALIGATOR_DEPRECATED void addConstraint(Cstr &&cstr);

  /// @copybrief  addConstraint().
  /// @details    Adds a constraint by allocating a new StageConstraintTpl.
  void addConstraint(const PolyFunction &func,
                     const PolyConstraintSet &cstr_set);

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
};

template <typename Scalar>
template <typename Cstr, typename>
void StageModelTpl<Scalar>::addConstraint(Cstr &&cstr) {
  const int c_nu = cstr.func->nu;
  if (c_nu != this->nu()) {
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}", c_nu,
        this->nu()));
  }
  constraints_.pushBack(std::forward<Cstr>(cstr));
}

} // namespace aligator

template <typename Scalar>
struct fmt::formatter<aligator::StageModelTpl<Scalar>>
    : fmt::ostream_formatter {};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/stage-model.txx"
#endif

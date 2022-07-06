#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>
#include <proxnlp/modelling/constraints/equality-constraint.hpp>

#include "proxddp/core/costs.hpp"
#include "proxddp/core/dynamics.hpp"
#include "proxddp/core/constraint.hpp"

#include "proxddp/core/clone.hpp"

namespace proxddp {

/** @brief    A stage in the control problem.
 *
 *  @details  Each stage containts cost functions, dynamical
 *            and constraint models.
 */
template <typename _Scalar>
class StageModelTpl : public cloneable<StageModelTpl<_Scalar>> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using Manifold = ManifoldAbstractTpl<Scalar>;
  using Dynamics = DynamicsModelTpl<Scalar>;
  using Constraint = StageConstraintTpl<Scalar>;
  using CostAbstract = CostAbstractTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;

  using ConstraintPtr = shared_ptr<Constraint>;
  using ManifoldPtr = shared_ptr<Manifold>;
  using CostPtr = shared_ptr<CostAbstract>;

  /// State space for the current state \f$x_k\f$.
  ManifoldPtr xspace_;
  /// State space for the next state \f$x_{k+1}\f$.
  ManifoldPtr xspace_next_;
  /// Control vector space -- by default, a simple Euclidean space.
  ManifoldPtr uspace_;

  CostPtr cost_;

  const Dynamics &dyn_model() const {
    return static_cast<const Dynamics &>(constraints_manager[0]->func());
  }
  const CostAbstract &cost() const { return *cost_; }

  ConstraintContainer<Scalar> constraints_manager;

  const Manifold &xspace() const { return *xspace_; }
  const Manifold &uspace() const { return *uspace_; }
  const Manifold &xspace_next() const { return *xspace_next_; }

  int nx1() const { return xspace_->nx(); }
  int ndx1() const { return xspace_->ndx(); }
  int nu() const { return uspace_->ndx(); }
  int nx2() const { return xspace_next_->nx(); }
  int ndx2() const { return xspace_next_->ndx(); }

  std::size_t numConstraints() const {
    return constraints_manager.numConstraints();
  }

  /// Number of primal optimization variables.
  int numPrimal() const;
  /// Number of dual variables, i.e. Lagrange multipliers.
  int numDual() const;

  /// Default constructor: assumes the control space is a Euclidean space of
  /// dimension \p nu.
  StageModelTpl(const ManifoldPtr &space1, const int nu,
                const ManifoldPtr &space2, const CostPtr &cost,
                const shared_ptr<Dynamics> &dyn_model);

  /// Secondary constructor: use a single manifold.
  StageModelTpl(const ManifoldPtr &space, const int nu, const CostPtr &cost,
                const shared_ptr<Dynamics> &dyn_model);

  /// @brief    Add a constraint to the stage.
  void addConstraint(const ConstraintPtr &cstr) {
    constraints_manager.push_back(cstr);
  }
  /// @copybrief addConstraint()
  void addConstraint(ConstraintPtr &&cstr) {
    constraints_manager.push_back(std::move(cstr));
  }
  /// @copybrief  addConstraint().
  /// @details    Adds a constraint by allocating a new StageConstraintTpl.
  void addConstraint(const shared_ptr<StageFunctionTpl<Scalar>> &func,
                     const shared_ptr<ConstraintSetBase<Scalar>> &cstr_set) {
    constraints_manager.push_back(std::make_shared<Constraint>(func, cstr_set));
  }

  /* Evaluate costs, constraints, ... */

  /// @brief    Evaluate all the functions (cost, dynamics, constraints) at this
  /// node.
  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const;

  /// @brief    Compute the derivatives of the StageModelTpl.
  void computeDerivatives(const ConstVectorRef &x, const ConstVectorRef &u,
                          const ConstVectorRef &y, Data &data) const;

  /// @brief    Create a Data object.
  shared_ptr<Data> createData() const { return std::make_shared<Data>(*this); }

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

/// @brief    Data struct for stage models StageModelTpl.
template <typename _Scalar>
struct StageDataTpl : public cloneable<StageDataTpl<_Scalar>> {
  using Scalar = _Scalar;
  PROXNLP_FUNCTION_TYPEDEFS(Scalar);

  using StageModel = StageModelTpl<Scalar>;
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;

  /// Data structs for the functions involved in the constraints.
  std::vector<shared_ptr<FunctionData>> constraint_data;
  /// Data struct for the dynamics.
  shared_ptr<DynamicsData> dyn_data() { return constraint_data[0]; }
  /// Data for the running costs.
  const shared_ptr<CostDataAbstract> cost_data;

  /// @brief    Constructor.
  ///
  /// @details  The constructor initializes or fills in the data members using
  /// move semantics.
  explicit StageDataTpl(const StageModel &stage_model);
};

} // namespace proxddp

#include "proxddp/core/stage-model.hxx"

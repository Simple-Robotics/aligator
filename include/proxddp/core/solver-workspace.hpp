#pragma once
#include "proxddp/fwd.hpp"

#include <fmt/format.h>
#include <ostream>

namespace proxddp {
namespace internal {

/// @brief  Contiguous storage for the value function parameters.
///
/// @details This provides storage for the matrix \f[
///     \begin{bmatrix} 2v & V_x^\top \\ V_x & V_{xx} \end{bmatrix}
/// \f]
template <typename Scalar> struct value_storage {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  MatrixXs storage;
  // Eigen::SelfAdjointView<MatrixXs, Eigen::Lower> store_sym_;
  Scalar &v_2() { return storage.coeffRef(0, 0); }
  VectorRef Vx_;
  MatrixRef Vxx_;

  value_storage(const int ndx)
      : storage(MatrixXs::Zero(ndx + 1, ndx + 1)),
        Vx_(storage.bottomRows(ndx).col(0)),
        Vxx_(storage.bottomRightCorner(ndx, ndx)) {}

  friend std::ostream &operator<<(std::ostream &oss,
                                  const value_storage &store) {
    Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "  [", "]");
    oss << "value_storage {\n";
    oss << store.storage.format(CleanFmt);
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
template <typename Scalar> struct q_function_storage {
protected:
  int ntot;

public:
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  MatrixXs storage;

  Scalar &q_2() { return storage.coeffRef(0, 0); }

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
      : ntot(ndx1 + nu + ndx2), storage(ntot + 1, ntot + 1),
        grad_(storage.bottomRows(ntot).col(0)),
        hess_(storage.bottomRightCorner(ntot, ntot)), Qx_(grad_.head(ndx1)),
        Qu_(grad_.segment(ndx1, nu)), Qy_(grad_.tail(ndx2)),
        Qxx_(hess_.topLeftCorner(ndx1, ndx1)),
        Qxu_(hess_.block(0, ndx1, ndx1, nu)),
        Qxy_(hess_.topRightCorner(ndx1, ndx2)),
        Quu_(hess_.block(ndx1, ndx1, nu, nu)),
        Quy_(hess_.block(ndx1, ndx1 + nu, nu, ndx2)),
        Qyy_(hess_.bottomRightCorner(ndx2, ndx2)) {
    storage.setZero();
  }
};

} // namespace internal

/** @brief Workspace for the solver.
 *
 * @details This struct holds data for the Riccati forward and backward passes,
 *          the primal-dual steps, problem data...
 */
template <typename _Scalar> struct WorkspaceTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using value_storage_t = internal::value_storage<Scalar>;
  using q_storage_t = internal::q_function_storage<Scalar>;

  const std::size_t nsteps;

  shared_ptr<TrajOptDataTpl<Scalar>> problem_data;

  /// Value function parameter storage
  std::vector<value_storage_t> value_params;

  /// Q-function storage
  std::vector<q_storage_t> q_params;

  std::vector<VectorXs> lams_plus_;
  std::vector<VectorXs> lams_pdal_;

  /// @name Riccati gains and buffers for primal-dual steps

  std::vector<VectorXs> pd_step_;
  std::vector<VectorRef> dxs_;
  std::vector<VectorRef> dus_;
  std::vector<VectorRef> dlams_;

  /// Buffer for KKT matrix
  MatrixXs kktMatrixFull_;
  /// Buffer for KKT right hand side
  MatrixXs kktRhsFull_;

  std::vector<VectorXs> trial_xs_;
  std::vector<VectorXs> trial_us_;
  std::vector<VectorXs> trial_lams_;

  /// @name Previous proximal iterates

  std::vector<VectorXs> prev_xs_;
  std::vector<VectorXs> prev_us_;
  std::vector<VectorXs> prev_lams_;

  /// Subproblem termination criterion for each stage.
  VectorXs inner_criterion_by_stage;
  /// Constraint violation for each stage of the TrajOptProblemTpl.
  VectorXs primal_infeas_by_stage;
  /// Dual infeasibility for each stage of the TrajOptProblemTpl.
  VectorXs dual_infeas_by_stage;

  /// Overall subproblem termination criterion.
  Scalar inner_criterion = 0.;

  explicit WorkspaceTpl(const TrajOptProblemTpl<Scalar> &problem);

  MatrixRef getKktView(const int nprim, const int ndual) {
    return kktMatrixFull_.topLeftCorner(nprim + ndual, nprim + ndual);
  }

  MatrixRef getKktRhs(const int nprim, const int ndual, const int ndx1) {
    return kktRhsFull_.topLeftCorner(nprim + ndual, ndx1 + 1);
  }

  friend std::ostream &operator<<(std::ostream &oss,
                                  const WorkspaceTpl<Scalar> &self) {
    oss << "Workspace {";
    oss << fmt::format("\n  num nodes      : {:d}", self.trial_us_.size())
        << fmt::format("\n  kkt buffer size: {:d}", self.kktMatrixFull_.rows());
    oss << "\n}";
    return oss;
  }
};

} // namespace proxddp

#include "proxddp/core/solver-workspace.hxx"

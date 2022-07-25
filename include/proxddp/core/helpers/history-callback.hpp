#pragma once

#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/solver-proxddp.hpp"

namespace proxddp {
namespace helpers {
/** @brief  Store the history of results.
 */
template <typename Scalar> struct history_callback : base_callback<Scalar> {
  history_callback(bool store_pd_vars = false, bool store_values = true,
                   bool store_residuals = true)
      : store_primal_dual_vars_(store_pd_vars), store_values_(store_values),
        store_residuals_(store_residuals) {}

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  struct {
    std::vector<std::vector<VectorXs>> xs;
    std::vector<std::vector<VectorXs>> lams;
    std::vector<Scalar> values;
    std::vector<Scalar> merit_values;
    std::vector<Scalar> prim_infeas;
    std::vector<Scalar> dual_infeas;
    std::vector<std::size_t> al_index;
    std::vector<Scalar> prim_tols;
    std::vector<Scalar> dual_tols;
  } storage;

  void call(const SolverProxDDP<Scalar> *solver,
            const WorkspaceTpl<Scalar> &workspace,
            const ResultsTpl<Scalar> &results) {
    if (store_primal_dual_vars_) {
      storage.xs.push_back(results.xs_);
      storage.lams.push_back(results.lams_);
    }
    if (store_values_) {
      storage.values.push_back(results.traj_cost_);
      storage.merit_values.push_back(results.merit_value_);
    }
    if (store_residuals_) {
      storage.prim_infeas.push_back(results.primal_infeasibility);
      storage.dual_infeas.push_back(results.dual_infeasibility);
    }
    storage.al_index.push_back(solver->al_iter);
    storage.prim_tols.push_back(solver->prim_tol_);
    storage.dual_tols.push_back(solver->inner_tol_);
  }

protected:
  const bool store_primal_dual_vars_;
  const bool store_values_;
  const bool store_residuals_;
};

} // namespace helpers
} // namespace proxddp

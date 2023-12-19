/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/callback-base.hpp"
#include "proxddp/core/workspace-base.hpp"
#include "proxddp/core/results-base.hpp"

namespace proxddp {

/// @brief  Store the history of results.
template <typename Scalar> struct HistoryCallbackTpl : CallbackBaseTpl<Scalar> {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  HistoryCallbackTpl(bool store_pd_vars = false, bool store_values = true,
                     bool store_residuals = true)
      : store_primal_dual_vars_(store_pd_vars), store_values_(store_values),
        store_residuals_(store_residuals) {}

  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);

  struct {
    std::vector<std::vector<VectorXs>> xs;
    std::vector<std::vector<VectorXs>> us;
    std::vector<std::vector<VectorXs>> lams;
    std::vector<Scalar> values;
    std::vector<Scalar> merit_values;
    std::vector<Scalar> prim_infeas;
    std::vector<Scalar> dual_infeas;
    std::vector<Scalar> inner_crits;
    std::vector<std::size_t> al_index;
    std::vector<Scalar> prim_tols;
    std::vector<Scalar> dual_tols;
  } storage;

  void call(const Workspace & /*workspace*/, const Results &results) {
    if (store_primal_dual_vars_) {
      storage.xs.push_back(results.xs);
      storage.us.push_back(results.us);
      storage.lams.push_back(results.lams);
    }
    if (store_values_) {
      storage.values.push_back(results.traj_cost_);
      storage.merit_values.push_back(results.merit_value_);
    }
    if (store_residuals_) {
      storage.prim_infeas.push_back(results.prim_infeas);
      storage.dual_infeas.push_back(results.dual_infeas);
    }
    // if (auto w = dynamic_cast<const WorkspaceTpl<Scalar> *>(&workspace)) {
    //   storage.inner_crits.push_back(w->inner_criterion);
    // }
    // if (auto r = dynamic_cast<const ResultsTpl<Scalar> *>(&results)) {
    //   storage.al_index.push_back(r->al_iter);
    // }
  }

  bool store_primal_dual_vars_;
  bool store_values_;
  bool store_residuals_;
};

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./history-callback.txx"
#endif

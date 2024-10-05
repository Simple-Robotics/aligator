/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/callback-base.hpp"
#include "aligator/core/workspace-base.hpp"
#include "aligator/core/results-base.hpp"

#include "aligator/solvers/proxddp/results.hpp"

namespace aligator {

/// @brief  Store the history of results.
template <typename Scalar> struct HistoryCallbackTpl : CallbackBaseTpl<Scalar> {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  HistoryCallbackTpl(bool store_pd_vars = false, bool store_values = true,
                     bool store_residuals = true)
      : store_primal_dual_vars_(store_pd_vars), store_values_(store_values),
        store_residuals_(store_residuals) {}

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

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

  void call(const Workspace & /*workspace*/, const Results &results) {
    if (store_primal_dual_vars_) {
      xs.push_back(results.xs);
      us.push_back(results.us);
      // lams.push_back(results.lams);
    }
    if (store_values_) {
      values.push_back(results.traj_cost_);
      merit_values.push_back(results.merit_value_);
    }
    if (store_residuals_) {
      prim_infeas.push_back(results.prim_infeas);
      dual_infeas.push_back(results.dual_infeas);
    }
    // if (auto w = dynamic_cast<const WorkspaceTpl<Scalar> *>(&workspace)) {
    //   inner_crits.push_back(w->inner_criterion);
    // }
    if (auto r = dynamic_cast<const ResultsTpl<Scalar> *>(&results)) {
      al_index.push_back(r->al_iter);
    }
  }

  bool store_primal_dual_vars_;
  bool store_values_;
  bool store_residuals_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./history-callback.txx"
#endif

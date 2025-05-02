/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/callback-base.hpp"
#include "aligator/solvers/workspace-base.hpp"
#include "aligator/solvers/results-base.hpp"

#include <typeindex>
#include <any>

namespace aligator {

/// @brief  Store the history of results.
template <typename Scalar> struct HistoryCallbackTpl : CallbackBaseTpl<Scalar> {
  using Workspace = WorkspaceBaseTpl<Scalar>;
  using Results = ResultsBaseTpl<Scalar>;
  template <typename Solver>
  HistoryCallbackTpl(Solver *solver, bool store_pd_vars = false,
                     bool store_values = true)
      : store_primal_dual_vars_(store_pd_vars), store_values_(store_values),
        rtti_(typeid(*solver)), solver_(solver) {}

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

  void call(const Workspace & /*workspace*/, const Results &results);

  bool store_primal_dual_vars_;
  bool store_values_;

private:
  std::type_index rtti_;
  std::any solver_;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./history-callback.txx"
#endif

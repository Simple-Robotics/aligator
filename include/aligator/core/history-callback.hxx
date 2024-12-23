#include "history-callback.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

namespace aligator {

template <typename Scalar>
void HistoryCallbackTpl<Scalar>::call(const Workspace & /*workspace*/,
                                      const Results &results) {
  if (store_primal_dual_vars_) {
    xs.push_back(results.xs);
    us.push_back(results.us);
    // lams.push_back(results.lams);
  }
  if (store_values_) {
    values.push_back(results.traj_cost_);
    merit_values.push_back(results.merit_value_);
  }
  prim_infeas.push_back(results.prim_infeas);
  dual_infeas.push_back(results.dual_infeas);
  // if (auto w = dynamic_cast<const WorkspaceTpl<Scalar> *>(&workspace)) {
  //   inner_crits.push_back(w->inner_criterion);
  // }
  if (auto r = dynamic_cast<const ResultsTpl<Scalar> *>(&results)) {
    al_index.push_back(r->al_iter);
  }

  if (rtti_ == typeid(SolverProxDDPTpl<Scalar>)) {
    auto *s = std::any_cast<SolverProxDDPTpl<Scalar> *>(solver_);
    prim_tols.push_back(s->prim_tol_);
    dual_tols.push_back(s->inner_tol_);
  }
}

} // namespace aligator

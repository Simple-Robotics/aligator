#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/action-model-wrap.hpp"
#include "proxddp/compat/crocoddyl/problem-wrap.hpp"
#include "proxddp/compat/crocoddyl/context.hpp"

namespace proxddp {
namespace compat {
namespace croc {

extern template
::proxddp::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(const boost::shared_ptr<context::CrocShootingProblem> &);

extern template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocCostModel>);

extern template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocActionModel>);

} // namespace croc
} // namespace compat
} // namespace proxddp

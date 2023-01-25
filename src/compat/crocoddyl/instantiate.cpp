/**
 * @file    instantiate.cpp
 * @details Instantiates the templates for the chosen context's Scalar type.
 */

#include "proxddp/compat/crocoddyl/instantiate.txx"

namespace proxddp {
namespace compat {
namespace croc {

template ::proxddp::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocCostModel>);

template CrocCostModelWrapperTpl<context::Scalar>::CrocCostModelWrapperTpl(
    boost::shared_ptr<context::CrocActionModel>);

} // namespace croc
} // namespace compat
} // namespace proxddp

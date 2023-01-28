/// @file    instantiate.cpp
/// @details Instantiates the templates for the chosen context's Scalar type.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

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

template DynamicsDataWrapperTpl<context::Scalar>::DynamicsDataWrapperTpl(
    const context::CrocActionModel *);

template CrocCostDataWrapperTpl<context::Scalar>::CrocCostDataWrapperTpl(
    const boost::shared_ptr<context::CrocCostData> &);

template CrocCostDataWrapperTpl<context::Scalar>::CrocCostDataWrapperTpl(
    const boost::shared_ptr<context::CrocActionData> &);

} // namespace croc
} // namespace compat
} // namespace proxddp

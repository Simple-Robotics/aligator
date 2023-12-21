/// @file    instantiate.cpp
/// @details Instantiates the templates for the chosen context's Scalar type.
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA

#include "aligator/compat/crocoddyl/instantiate.txx"

namespace aligator {
namespace compat {
namespace croc {

template ::aligator::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

template struct StateWrapperTpl<context::Scalar>;

template struct CrocCostModelWrapperTpl<context::Scalar>;
template struct CrocCostDataWrapperTpl<context::Scalar>;

template struct DynamicsDataWrapperTpl<context::Scalar>;

template struct ActionModelWrapperTpl<context::Scalar>;
template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace croc
} // namespace compat
} // namespace aligator

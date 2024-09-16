/// @file    instantiate.cpp
/// @details Instantiates the templates for the chosen context's Scalar type.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA

#include "aligator/compat/crocoddyl/instantiate.txx"

namespace aligator::compat::croc {

template ::aligator::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

template struct StateWrapperTpl<context::Scalar>;

template struct DynamicsDataWrapperTpl<context::Scalar>;

} // namespace aligator::compat::croc

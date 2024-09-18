/// @file    instantiate.cpp
/// @details Instantiates the templates for the chosen context's Scalar type.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#include "aligator/compat/crocoddyl/problem-wrap.hxx"

namespace aligator::compat::croc {

template ::aligator::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

} // namespace aligator::compat::croc

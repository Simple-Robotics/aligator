/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "context.hpp"
#include <crocoddyl/core/optctrl/shooting.hpp>

namespace aligator::compat::croc {

///
/// @brief   This function converts a crocoddyl::ShootingProblemTpl
/// to an aligator::TrajOptProblemTpl
template <typename Scalar>
TrajOptProblemTpl<Scalar> convertCrocoddylProblem(
    const boost::shared_ptr<crocoddyl::ShootingProblemTpl<Scalar>>
        &croc_problem);

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template ::aligator::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);
#endif

} // namespace aligator::compat::croc

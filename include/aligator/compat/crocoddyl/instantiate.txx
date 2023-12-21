/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/compat/crocoddyl/action-model-wrap.hpp"
#include "aligator/compat/crocoddyl/context.hpp"
#include "aligator/compat/crocoddyl/cost-wrap.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"

namespace aligator {
namespace compat {
namespace croc {

extern template ::aligator::context::TrajOptProblem
convertCrocoddylProblem<context::Scalar>(
    const boost::shared_ptr<context::CrocShootingProblem> &);

extern template struct StateWrapperTpl<context::Scalar>;

extern template struct CrocCostModelWrapperTpl<context::Scalar>;
extern template struct CrocCostDataWrapperTpl<context::Scalar>;

extern template struct DynamicsDataWrapperTpl<context::Scalar>;

extern template struct ActionModelWrapperTpl<context::Scalar>;
extern template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace croc
} // namespace compat
} // namespace aligator

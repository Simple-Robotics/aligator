/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA

#include "aligator/core/traj-opt-problem.hpp"

#include <Eigen/Core>

#include <pinocchio/multibody/fwd.hpp>

#include "aligator/modelling/dynamics/multibody-constraint-fwd.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/multibody/frame-velocity.hpp"
#include "aligator/modelling/multibody/frame-placement.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"

namespace pin = pinocchio;

using aligator::context::TrajOptData;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Model = pin::ModelTpl<double, 0>;
using ProximalSettings = pin::ProximalSettingsTpl<double>;
using MultibodyConstraintFwdDynamics =
    aligator::dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
using ODEAbstract = aligator::dynamics::ODEAbstractTpl<double>;
using MultibodyPhaseSpace = aligator::MultibodyPhaseSpace<double>;
using IntegratorSemiImplEuler =
    aligator::dynamics::IntegratorSemiImplEulerTpl<double>;
using StageModel = aligator::StageModelTpl<double>;
using FrameVelocityResidual = aligator::FrameVelocityResidualTpl<double>;
using FramePlacementResidual = aligator::FramePlacementResidualTpl<double>;
using CostStack = aligator::CostStackTpl<double>;
using QuadraticStateCost = aligator::QuadraticStateCostTpl<double>;
using QuadraticControlCost = aligator::QuadraticControlCostTpl<double>;
using QuadraticResidualCost = aligator::QuadraticResidualCostTpl<double>;
using TrajOptProblem = aligator::TrajOptProblemTpl<double>;

enum Support { LEFT, RIGHT, DOUBLE };

void makeTalosReduced(Model &model_complete, Model &model, Eigen::VectorXd &q0);

TrajOptProblem defineLocomotionProblem(const std::size_t T_ss,
                                       const std::size_t T_ds);

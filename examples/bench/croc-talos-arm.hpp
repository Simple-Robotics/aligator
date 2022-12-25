/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <example-robot-data/path.hpp>

#include "proxddp/compat/crocoddyl/problem-wrap.hpp"

namespace pin = pinocchio;
namespace croc = crocoddyl;

inline void makeTalosArm(pin::Model &model) {
  const std::string talos_arm_path =
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/talos_data/robots/talos_left_arm.urdf";
  pin::urdf::buildModel(talos_arm_path, model);
}

boost::shared_ptr<croc::ShootingProblem>
defineCrocoddylProblem(std::size_t nsteps);

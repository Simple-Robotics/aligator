#include <Eigen/Core>
#include <pinocchio/multibody/sample-models.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "aligator/modelling/spaces/multibody.hpp"

#include <catch2/catch_test_macros.hpp>

#include <aligator/modelling/multibody/contact-force.hpp>
#include <aligator/modelling/multibody/multibody-wrench-cone.hpp>
#include <aligator/modelling/multibody/multibody-friction-cone.hpp>

TEST_CASE("contact_forces_6d", "[forces]") {
  using namespace Eigen;
  using namespace pinocchio;
  using namespace aligator;

  using ContactForceData = aligator::ContactForceDataTpl<double>;
  using ContactForceResidual = aligator::ContactForceResidualTpl<double>;
  using StageFunctionData = aligator::StageFunctionDataTpl<double>;
  using Manifold = aligator::MultibodyPhaseSpace<double>;
  using Vector3or6 = Eigen::Matrix<double, -1, 1, Eigen::ColMajor, 6, 1>;

  Model model;
  buildModels::humanoidRandom(model, true);
  Data data(model), data_fd(model);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd x0(model.nv * 2 + 1);
  x0 << q, v;
  VectorXd u0 = VectorXd::Random(model.nv - 6);

  const std::string LF = "lleg6_joint";
  const Model::JointIndex LF_id = model.getJointId(LF);
  const std::string RF = "rleg6_joint";
  const Model::JointIndex RF_id = model.getJointId(RF);

  // Contact models and data
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintModel)
  constraint_models;
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData)
  constraint_data;

  RigidConstraintModel ci_LF(CONTACT_6D, model, LF_id, LOCAL);
  ci_LF.joint1_placement.setRandom();
  ci_LF.corrector.Kp.array() = 10;
  ci_LF.corrector.Kd.array() = 10;
  ci_LF.name = "LF_foot";

  RigidConstraintModel ci_RF(CONTACT_6D, model, RF_id, LOCAL);
  ci_RF.joint1_placement.setRandom();
  ci_RF.corrector.Kp.array() = 10;
  ci_RF.corrector.Kd.array() = 10;
  ci_RF.name = "RF_foot";

  constraint_models.push_back(ci_LF);
  constraint_data.push_back(RigidConstraintData(ci_LF));
  constraint_models.push_back(ci_RF);
  constraint_data.push_back(RigidConstraintData(ci_RF));

  const double mu0 = 0.;
  ProximalSettings prox_settings(1e-12, mu0, 1);

  Manifold space{model};
  MatrixXd act_matrix(model.nv, model.nv - 6);
  act_matrix.setZero();
  act_matrix.bottomRows(model.nv - 6).setIdentity();

  // Test force 6d
  Vector3or6 fref{6};
  fref.setZero();
  fref[2] = 100;
  ContactForceResidual fun =
      ContactForceResidual(space.ndx(), model, act_matrix, constraint_models,
                           prox_settings, fref, "RF_foot");

  REQUIRE(fun.getReference().isApprox(fref));
  shared_ptr<StageFunctionData> sfdata = fun.createData();
  shared_ptr<ContactForceData> fdata =
      std::static_pointer_cast<ContactForceData>(sfdata);

  forwardKinematics(model, fdata->pin_data_, q);
  updateFramePlacements(model, data);
  fun.evaluate(x0, u0, *fdata);
  fun.computeJacobians(x0, u0, *fdata);

  MatrixXd lambda_partial_dx(6, model.nv * 2);
  MatrixXd lambda_partial_du(6, model.nv - 6);
  lambda_partial_dx = fdata->Jx_;
  lambda_partial_du = fdata->Ju_;

  // Data_fd
  MatrixXd lambda_partial_dx_fd(6, model.nv * 2);
  lambda_partial_dx_fd.setZero();
  MatrixXd lambda_partial_du_fd(6, model.nv - 6);
  lambda_partial_du_fd.setZero();

  const VectorXd lambda0 = fdata->value_;
  VectorXd v_eps(VectorXd::Zero(model.nv * 2));
  VectorXd u_eps(VectorXd::Zero(model.nv - 6));
  VectorXd x_plus(model.nq + model.nv);
  VectorXd u_plus(model.nv - 6);

  const double alpha = 1e-8;

  for (int k = 0; k < model.nv * 2; ++k) {
    v_eps[k] += alpha;
    x_plus = space.integrate(x0, v_eps);
    fun.evaluate(x_plus, u0, *fdata);
    lambda_partial_dx_fd.col(k) = (fdata->value_ - lambda0) / alpha;
    v_eps[k] = 0.;
  }

  for (int k = 0; k < model.nv - 6; ++k) {
    u_eps[k] += alpha;
    u_plus = u0 + u_eps;
    fun.evaluate(x0, u_plus, *fdata);
    lambda_partial_du_fd.col(k) = (fdata->value_ - lambda0) / alpha;
    u_eps[k] = 0.;
  }
  // std::cout << sqrt(alpha) << std::endl;
  REQUIRE(lambda_partial_dx_fd.isApprox(lambda_partial_dx, sqrt(alpha)));
  REQUIRE(lambda_partial_du_fd.isApprox(lambda_partial_du, sqrt(alpha)));
}

TEST_CASE("contact_forces_3d", "[forces]") {
  using namespace Eigen;
  using namespace pinocchio;
  using namespace aligator;

  using ContactForceData = aligator::ContactForceDataTpl<double>;
  using ContactForceResidual = aligator::ContactForceResidualTpl<double>;
  using StageFunctionData = aligator::StageFunctionDataTpl<double>;
  using Manifold = aligator::MultibodyPhaseSpace<double>;
  using Vector3or6 = Eigen::Matrix<double, -1, 1, Eigen::ColMajor, 6, 1>;

  Model model;
  buildModels::humanoidRandom(model, true);
  Data data(model), data_fd(model);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd x0(model.nv * 2 + 1);
  x0 << q, v;
  VectorXd u0 = VectorXd::Random(model.nv - 6);

  const std::string LF = "lleg6_joint";
  const Model::JointIndex LF_id = model.getJointId(LF);
  const std::string RF = "rleg6_joint";
  const Model::JointIndex RF_id = model.getJointId(RF);

  // Contact models and data
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintModel)
  constraint_models;
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData)
  constraint_data;

  RigidConstraintModel ci_LF(CONTACT_3D, model, LF_id, LOCAL);
  ci_LF.joint1_placement.setRandom();
  ci_LF.corrector.Kp.array() = 10;
  ci_LF.corrector.Kd.array() = 10;
  ci_LF.name = "LF_foot";

  RigidConstraintModel ci_RF(CONTACT_3D, model, RF_id, LOCAL);
  ci_RF.joint1_placement.setRandom();
  ci_RF.corrector.Kp.array() = 10;
  ci_RF.corrector.Kd.array() = 10;
  ci_RF.name = "RF_foot";

  constraint_models.push_back(ci_LF);
  constraint_data.push_back(RigidConstraintData(ci_LF));
  constraint_models.push_back(ci_RF);
  constraint_data.push_back(RigidConstraintData(ci_RF));

  const double mu0 = 0.;
  ProximalSettings prox_settings(1e-12, mu0, 1);

  Manifold space{model};
  MatrixXd act_matrix(model.nv, model.nv - 6);
  act_matrix.setZero();
  act_matrix.bottomRows(model.nv - 6).setIdentity();

  // Test force 3d
  Vector3or6 fref{3};
  fref.setZero();
  fref[2] = 100;
  ContactForceResidual fun =
      ContactForceResidual(space.ndx(), model, act_matrix, constraint_models,
                           prox_settings, fref, "RF_foot");
  shared_ptr<StageFunctionData> sfdata = fun.createData();
  shared_ptr<ContactForceData> fdata =
      std::static_pointer_cast<ContactForceData>(sfdata);

  REQUIRE(fun.getReference().isApprox(fref));

  forwardKinematics(model, fdata->pin_data_, q);
  updateFramePlacements(model, data);
  fun.evaluate(x0, u0, *fdata);
  fun.computeJacobians(x0, u0, *fdata);

  MatrixXd lambda_partial_dx(3, model.nv * 2);
  MatrixXd lambda_partial_du(3, model.nv - 6);
  lambda_partial_dx = fdata->Jx_;
  lambda_partial_du = fdata->Ju_;

  // Data_fd
  MatrixXd lambda_partial_dx_fd(3, model.nv * 2);
  lambda_partial_dx_fd.setZero();
  MatrixXd lambda_partial_du_fd(3, model.nv - 6);
  lambda_partial_du_fd.setZero();

  const VectorXd lambda0 = fdata->value_;
  VectorXd v_eps(VectorXd::Zero(model.nv * 2));
  VectorXd u_eps(VectorXd::Zero(model.nv - 6));
  VectorXd x_plus(model.nq + model.nv);
  VectorXd u_plus(model.nv - 6);

  const double alpha = 1e-8;

  for (int k = 0; k < model.nv * 2; ++k) {
    v_eps[k] += alpha;
    x_plus = space.integrate(x0, v_eps);
    fun.evaluate(x_plus, u0, *fdata);
    lambda_partial_dx_fd.col(k) = (fdata->value_ - lambda0) / alpha;
    v_eps[k] = 0.;
  }

  for (int k = 0; k < model.nv - 6; ++k) {
    u_eps[k] += alpha;
    u_plus = u0 + u_eps;
    fun.evaluate(x0, u_plus, *fdata);
    lambda_partial_du_fd.col(k) = (fdata->value_ - lambda0) / alpha;
    u_eps[k] = 0.;
  }
  // std::cout << sqrt(alpha) << std::endl;
  REQUIRE(lambda_partial_dx_fd.isApprox(lambda_partial_dx, sqrt(alpha)));
  REQUIRE(lambda_partial_du_fd.isApprox(lambda_partial_du, sqrt(alpha)));
}

TEST_CASE("wrench_cone", "[forces]") {
  using namespace Eigen;
  using namespace pinocchio;
  using namespace aligator;

  using MultibodyWrenchConeData = aligator::MultibodyWrenchConeDataTpl<double>;
  using MultibodyWrenchConeResidual =
      aligator::MultibodyWrenchConeResidualTpl<double>;
  using StageFunctionData = aligator::StageFunctionDataTpl<double>;
  using Manifold = aligator::MultibodyPhaseSpace<double>;

  Model model;
  buildModels::humanoidRandom(model, true);
  Data data(model), data_fd(model);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd x0(model.nv * 2 + 1);
  x0 << q, v;
  VectorXd u0 = VectorXd::Random(model.nv - 6);

  const std::string LF = "lleg6_joint";
  const Model::JointIndex LF_id = model.getJointId(LF);
  const std::string RF = "rleg6_joint";
  const Model::JointIndex RF_id = model.getJointId(RF);

  // Contact models and data
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintModel)
  constraint_models;
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData)
  constraint_data;

  RigidConstraintModel ci_LF(CONTACT_6D, model, LF_id, LOCAL);
  ci_LF.joint1_placement.setRandom();
  ci_LF.corrector.Kp.array() = 10;
  ci_LF.corrector.Kd.array() = 10;
  ci_LF.name = "LF_foot";

  RigidConstraintModel ci_RF(CONTACT_6D, model, RF_id, LOCAL);
  ci_RF.joint1_placement.setRandom();
  ci_RF.corrector.Kp.array() = 10;
  ci_RF.corrector.Kd.array() = 10;
  ci_RF.name = "RF_foot";

  constraint_models.push_back(ci_LF);
  constraint_data.push_back(RigidConstraintData(ci_LF));
  // constraint_models.push_back(ci_RF);
  // constraint_data.push_back(RigidConstraintData(ci_RF));

  const double mu0 = 0.;
  ProximalSettings prox_settings(1e-12, mu0, 1);

  Manifold space{model};
  MatrixXd act_matrix(model.nv, model.nv - 6);
  act_matrix.setZero();
  act_matrix.bottomRows(model.nv - 6).setIdentity();

  double mu = 0.1;
  double hL = 0.2;
  double hW = 0.2;
  MultibodyWrenchConeResidual fun = MultibodyWrenchConeResidual(
      space.ndx(), model, act_matrix, constraint_models, prox_settings,
      "LF_foot", mu, hL, hW);
  shared_ptr<StageFunctionData> sfdata = fun.createData();
  shared_ptr<MultibodyWrenchConeData> fdata =
      std::static_pointer_cast<MultibodyWrenchConeData>(sfdata);

  forwardKinematics(model, fdata->pin_data_, q);
  updateFramePlacements(model, data);
  fun.evaluate(x0, u0, *fdata);
  fun.computeJacobians(x0, u0, *fdata);

  MatrixXd cone_partial_dx(17, model.nv * 2);
  MatrixXd cone_partial_du(17, model.nv - 6);
  cone_partial_dx = fdata->Jx_;
  cone_partial_du = fdata->Ju_;

  // Data_fd
  MatrixXd cone_partial_dx_fd(17, model.nv * 2);
  cone_partial_dx_fd.setZero();
  MatrixXd cone_partial_du_fd(17, model.nv - 6);
  cone_partial_du_fd.setZero();

  const VectorXd cone0 = fdata->value_;
  VectorXd v_eps(VectorXd::Zero(model.nv * 2));
  VectorXd u_eps(VectorXd::Zero(model.nv - 6));
  VectorXd x_plus(model.nq + model.nv);
  VectorXd u_plus(model.nv - 6);

  const double alpha = 1e-8;

  for (int k = 0; k < model.nv * 2; ++k) {
    v_eps[k] += alpha;
    x_plus = space.integrate(x0, v_eps);
    fun.evaluate(x_plus, u0, *fdata);
    cone_partial_dx_fd.col(k) = (fdata->value_ - cone0) / alpha;
    v_eps[k] = 0.;
  }

  for (int k = 0; k < model.nv - 6; ++k) {
    u_eps[k] += alpha;
    u_plus = u0 + u_eps;
    fun.evaluate(x0, u_plus, *fdata);
    cone_partial_du_fd.col(k) = (fdata->value_ - cone0) / alpha;
    u_eps[k] = 0.;
  }

  REQUIRE(cone_partial_dx_fd.isApprox(cone_partial_dx, sqrt(alpha)));
  REQUIRE(cone_partial_du_fd.isApprox(cone_partial_du, sqrt(alpha)));
}

TEST_CASE("friction_cone", "[forces]") {
  using namespace Eigen;
  using namespace pinocchio;
  using namespace aligator;

  using MultibodyFrictionConeData =
      aligator::MultibodyFrictionConeDataTpl<double>;
  using MultibodyFrictionConeResidual =
      aligator::MultibodyFrictionConeResidualTpl<double>;
  using StageFunctionData = aligator::StageFunctionDataTpl<double>;
  using Manifold = aligator::MultibodyPhaseSpace<double>;

  Model model;
  buildModels::humanoidRandom(model, true);
  Data data(model), data_fd(model);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd x0(model.nv * 2 + 1);
  x0 << q, v;
  VectorXd u0 = VectorXd::Random(model.nv - 6);

  const std::string LF = "lleg6_joint";
  const Model::JointIndex LF_id = model.getJointId(LF);
  const std::string RF = "rleg6_joint";
  const Model::JointIndex RF_id = model.getJointId(RF);

  // Contact models and data
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintModel)
  constraint_models;
  PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData)
  constraint_data;

  RigidConstraintModel ci_LF(CONTACT_3D, model, LF_id, LOCAL);
  ci_LF.joint1_placement.setRandom();
  ci_LF.corrector.Kp.array() = 10;
  ci_LF.corrector.Kd.array() = 10;
  ci_LF.name = "LF_foot";

  RigidConstraintModel ci_RF(CONTACT_3D, model, RF_id, LOCAL);
  ci_RF.joint1_placement.setRandom();
  ci_RF.corrector.Kp.array() = 10;
  ci_RF.corrector.Kd.array() = 10;
  ci_RF.name = "RF_foot";

  constraint_models.push_back(ci_LF);
  constraint_data.push_back(RigidConstraintData(ci_LF));
  // constraint_models.push_back(ci_RF);
  // constraint_data.push_back(RigidConstraintData(ci_RF));

  const double mu0 = 0.;
  ProximalSettings prox_settings(1e-12, mu0, 1);

  Manifold space{model};
  MatrixXd act_matrix(model.nv, model.nv - 6);
  act_matrix.setZero();
  act_matrix.bottomRows(model.nv - 6).setIdentity();

  double mu = 0.1;
  MultibodyFrictionConeResidual fun = MultibodyFrictionConeResidual(
      space.ndx(), model, act_matrix, constraint_models, prox_settings,
      "LF_foot", mu);
  shared_ptr<StageFunctionData> sfdata = fun.createData();
  shared_ptr<MultibodyFrictionConeData> fdata =
      std::static_pointer_cast<MultibodyFrictionConeData>(sfdata);

  forwardKinematics(model, fdata->pin_data_, q);
  updateFramePlacements(model, data);
  fun.evaluate(x0, u0, *fdata);
  fun.computeJacobians(x0, u0, *fdata);

  MatrixXd cone_partial_dx(2, model.nv * 2);
  MatrixXd cone_partial_du(2, model.nv - 6);
  cone_partial_dx = fdata->Jx_;
  cone_partial_du = fdata->Ju_;

  // Data_fd
  MatrixXd cone_partial_dx_fd(2, model.nv * 2);
  cone_partial_dx_fd.setZero();
  MatrixXd cone_partial_du_fd(2, model.nv - 6);
  cone_partial_du_fd.setZero();

  const VectorXd cone0 = fdata->value_;
  VectorXd v_eps(VectorXd::Zero(model.nv * 2));
  VectorXd u_eps(VectorXd::Zero(model.nv - 6));
  VectorXd x_plus(model.nq + model.nv);
  VectorXd u_plus(model.nv - 6);

  const double alpha = 1e-8;

  for (int k = 0; k < model.nv * 2; ++k) {
    v_eps[k] += alpha;
    x_plus = space.integrate(x0, v_eps);
    fun.evaluate(x_plus, u0, *fdata);
    cone_partial_dx_fd.col(k) = (fdata->value_ - cone0) / alpha;
    v_eps[k] = 0.;
  }

  for (int k = 0; k < model.nv - 6; ++k) {
    u_eps[k] += alpha;
    u_plus = u0 + u_eps;
    fun.evaluate(x0, u_plus, *fdata);
    cone_partial_du_fd.col(k) = (fdata->value_ - cone0) / alpha;
    u_eps[k] = 0.;
  }

  REQUIRE(cone_partial_dx_fd.isApprox(cone_partial_dx, sqrt(alpha)));
  REQUIRE(cone_partial_du_fd.isApprox(cone_partial_du, sqrt(alpha)));
}

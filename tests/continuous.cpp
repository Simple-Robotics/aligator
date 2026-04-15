#include <catch2/catch_test_macros.hpp>

#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include <pinocchio/multibody/sample-models.hpp>
#endif

#ifdef ALIGATOR_WITH_PINOCCHIO
using namespace aligator;

TEST_CASE("create_data", "[continuous]") {
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model);

  using StateMultibody = aligator::MultibodyPhaseSpace<double>;
  const StateMultibody state{model};
  Eigen::MatrixXd B(model.nv, model.nv);
  B.setIdentity();
  dynamics::MultibodyFreeFwdDynamicsTpl<double> contdyn(state, B);

  using ContDataAbstract = dynamics::ContinuousDynamicsDataTpl<double>;
  using Data = dynamics::MultibodyFreeFwdDataTpl<double>;
  shared_ptr<ContDataAbstract> data = contdyn.createData();
  shared_ptr<Data> d2 = std::static_pointer_cast<Data>(data);

  REQUIRE(d2->tau_.size() == model.nv);
  REQUIRE(d2->dtau_du_.cols() == model.nv);
  REQUIRE(d2->dtau_du_.rows() == model.nv);
}

#endif

#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include <pinocchio/parsers/sample-models.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(continuous)

using namespace aligator;

BOOST_AUTO_TEST_CASE(create_data) {
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model);

  using StateMultibody = proxsuite::nlp::MultibodyPhaseSpace<double>;
  auto spaceptr = std::make_shared<StateMultibody>(model);
  Eigen::MatrixXd B(model.nv, model.nv);
  B.setIdentity();
  dynamics::MultibodyFreeFwdDynamicsTpl<double> contdyn(spaceptr, B);

  using ContDataAbstract = dynamics::ContinuousDynamicsDataTpl<double>;
  using Data = dynamics::MultibodyFreeFwdDataTpl<double>;
  using CommonModelBuilderContainer = CommonModelBuilderContainerTpl<double>;
  CommonModelBuilderContainer builder_container;
  contdyn.configure(builder_container);
  auto model_container = builder_container.createCommonModelContainer();
  auto data_container = model_container.createData();
  shared_ptr<ContDataAbstract> data = contdyn.createData(data_container);
  shared_ptr<Data> d2 = std::static_pointer_cast<Data>(data);

  BOOST_CHECK_EQUAL(d2->multibody_data_->tau_.size(), model.nv);
  BOOST_CHECK_EQUAL(d2->multibody_data_->qdd_dtau_.cols(), model.nv);
  BOOST_CHECK_EQUAL(d2->multibody_data_->qdd_dtau_.rows(), model.nv);
}

BOOST_AUTO_TEST_SUITE_END()

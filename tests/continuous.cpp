#include <boost/test/unit_test.hpp>

#ifdef ALIGATOR_WITH_PINOCCHIO
#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"
#include <pinocchio/parsers/sample-models.hpp>
#endif

BOOST_AUTO_TEST_SUITE(continuous)

#ifdef ALIGATOR_WITH_PINOCCHIO
using namespace aligator;

BOOST_AUTO_TEST_CASE(create_data) {
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model);

  using StateMultibody = proxsuite::nlp::MultibodyPhaseSpace<double>;
  const StateMultibody state{model};
  Eigen::MatrixXd B(model.nv, model.nv);
  B.setIdentity();
  dynamics::MultibodyFreeFwdDynamicsTpl<double> contdyn(state, B);

  using ContDataAbstract = dynamics::ContinuousDynamicsDataTpl<double>;
  using Data = dynamics::MultibodyFreeFwdDataTpl<double>;
  shared_ptr<ContDataAbstract> data = contdyn.createData();
  shared_ptr<Data> d2 = std::static_pointer_cast<Data>(data);

  BOOST_CHECK_EQUAL(d2->tau_.size(), model.nv);
  BOOST_CHECK_EQUAL(d2->dtau_du_.cols(), model.nv);
  BOOST_CHECK_EQUAL(d2->dtau_du_.rows(), model.nv);
}

#endif

BOOST_AUTO_TEST_SUITE_END()

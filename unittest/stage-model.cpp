#include "proxddp/core/stage-model.hpp"
#include "proxddp/utils.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"

#include <boost/test/unit_test.hpp>


#include <fmt/core.h>
#include <fmt/ostream.h>


BOOST_AUTO_TEST_SUITE(node)

using namespace proxddp;

using Scalar = double;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct AddModel : ExplicitDynamicsModelTpl<double>
{
  AddModel(const Manifold& space, const int nu)
    : ExplicitDynamicsModelTpl<double>(space, nu) {}  
  void forward(const ConstVectorRef& x, const ConstVectorRef& u, VectorRef out) const
  {
    out_space_.integrate(x, u, out);
  }

  void dForward(const ConstVectorRef& x, const ConstVectorRef& u, MatrixRef Jx, MatrixRef Ju) const
  {
    out_space_.Jintegrate(x, u, Jx, 0);
    out_space_.Jintegrate(x, u, Ju, 1);
  }
};


BOOST_AUTO_TEST_CASE(test_node1)
{
  using Manifold = proxnlp::VectorSpaceTpl<Scalar>;
  using Stage = StageModelTpl<Scalar>;

  constexpr int NX = 4;
  constexpr int NU = NX;

  Manifold space(NX);
  AddModel dyn_model(space, NU);
  Stage stage(space, NU, dyn_model);

  fmt::print("Node: {}\n", stage);

  Eigen::VectorXd u0(NU);
  u0.setZero();
  auto x0 = space.rand();
  constexpr int nsteps = 20;
  std::vector<Eigen::VectorXd> us(nsteps, u0);

  auto xs = rollout(dyn_model, x0, us);
  for (std::size_t i = 0; i < xs.size(); i++)
  {
    BOOST_CHECK(x0.isApprox(xs[i]));
  }

}


BOOST_AUTO_TEST_SUITE_END()

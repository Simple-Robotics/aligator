#include "proxddp/core/stage-model.hpp"

#include "proxnlp/modelling/spaces/vector-space.hpp"

#include <boost/test/unit_test.hpp>


#include <fmt/core.h>
#include <fmt/ostream.h>


BOOST_AUTO_TEST_SUITE(node)

using namespace proxddp;

using Scalar = double;

/// @brief    Constant dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x \f$ itself.
struct ConstantModel : DynamicsModelTpl<double>
{
  ConstantModel(const int ndx, const int nu)
    : DynamicsModelTpl<double>(ndx, nu) {}  
  void evaluate(const ConstVectorRef& x, const ConstVectorRef&, const ConstVectorRef& y, Data& data) const
  {
    data.value_ = y - x;
  }

  void computeJacobians(const ConstVectorRef&, const ConstVectorRef&, const ConstVectorRef&, Data& data) const
  {
    data.Jx_.setIdentity();
    data.Jx_ *= -1.;
    data.Ju_.setZero();
    data.Jy_.setIdentity();
  }
};


BOOST_AUTO_TEST_CASE(test_node1)
{
  using Manifold = proxnlp::VectorSpaceTpl<Scalar>;
  using Stage = StageModelTpl<Scalar>;

  constexpr int NX = 4;
  constexpr int NU = 2;

  Manifold space(NX);
  ConstantModel dynModel(space.ndx(), NU);
  Stage node(space, NU, dynModel);

  fmt::print("Node: {}", node);

}


BOOST_AUTO_TEST_SUITE_END()

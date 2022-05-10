/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/utils.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>

#include "boost/optional.hpp"

using namespace proxddp;


template<typename _Scalar>
struct LinearDiscreteDynamics : ExplicitDynamicsModelTpl<_Scalar>
{
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  const MatrixXs A_;
  const MatrixXs B_;
  VectorXs c_;

  using Base = ExplicitDynamicsModelTpl<double>;
  LinearDiscreteDynamics(const MatrixXs& A,
                         const MatrixXs& B,
                         const boost::optional<VectorXs>& c = boost::none)
    : Base(std::make_shared<proxnlp::VectorSpaceTpl<double>>((int)A.cols()), (int)B.cols())
    , A_(A), B_(B)
    {
      if (boost::optional<VectorXs> value = c)
      {
        c_ = *value;
      } else {
        c_ = VectorXs::Zero(A.cols());
      }
    }

  void forward(const ConstVectorRef& x,
               const ConstVectorRef& u,
               VectorRef out) const override
  {
    out = A_ * x + B_ * u + c_;
  }
  
  void dForward(const ConstVectorRef&,
                const ConstVectorRef&,
                MatrixRef Jx, MatrixRef Ju) const override
  {
    Jx = A_;
    Ju = B_;
  }

};


int main()
{

  const int dim = 3;
  const int nu = 2;
  Eigen::MatrixXd A(dim, dim);
  Eigen::MatrixXd B(dim, nu);
  A.setIdentity();
  B.setRandom();

  LinearDiscreteDynamics<double> dynamics(A, B);

  auto x0 = dynamics.out_space().rand();
  std::size_t nsteps = 10;
  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < nsteps; i++)
  {
    us.push_back(Eigen::VectorXd::Random(nu));
  }
  auto xs = rollout(dynamics, x0, us);

  return 0;
}


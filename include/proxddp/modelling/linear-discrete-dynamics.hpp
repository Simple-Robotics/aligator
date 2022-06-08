#pragma once


#include "proxddp/core/explicit-dynamics.hpp"
#include <proxnlp/modelling/spaces/vector-space.hpp>

#include "boost/optional.hpp"


namespace proxddp
{

  /// @brief Discrete explicit linear dynamics.
  template<typename _Scalar>
  struct LinearDiscreteDynamics : ExplicitDynamicsModelTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    const MatrixXs A_;
    const MatrixXs B_;
    const VectorXs c_;

    using Base = ExplicitDynamicsModelTpl<Scalar>;
    using VectorSpaceType = proxnlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

    /// @brief Constructor with state manifold and matrices.
    LinearDiscreteDynamics(
      const MatrixXs& A,
      const MatrixXs& B,
      const VectorXs& c)
      : Base(std::make_shared<VectorSpaceType>((int)A.cols()), (int)B.cols())
      , A_(A)
      , B_(B)
      , c_(c)
      {}

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
  
} // namespace proxddp


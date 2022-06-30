#pragma once

#include "proxddp/modelling/dynamics/ode-abstract.hpp"

#include <proxnlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/data.hpp>


namespace proxddp
{
  namespace dynamics
  {
    template<typename Scalar>
    struct MultibodyFreeFwdDataTpl;

    /**
     * @brief   Free-space multibody forward dynamics, using Pinocchio.
     * 
     * @details This is described in state-space \f$\mathcal{X} = T\mathcal{Q}\f$ (the phase space \f$x = (q,v)\f$)
     *          using the differential equation
     *          \f[
     *            \dot{x} = f(x, u) = \begin{bmatrix}
     *              v \\ a(q, v, \tau(u))
     *            \end{bmatrix}
     *          \f]
     *          where \f$\tau(u) = Bu\f$, \f$B\f$ is a given actuation matrix, and \f$a(q,v,\tau)\f$ is the acceleration
     *          computed from the ABA algorithm.
     * 
     */
    template<typename _Scalar>
    struct MultibodyFreeFwdDynamicsTpl : ODEAbstractTpl<_Scalar>
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      using Scalar = _Scalar;
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
      using Base = ODEAbstractTpl<Scalar>;
      using ODEData = ODEDataTpl<Scalar>;
      using ContDataAbstract = ContinuousDynamicsDataTpl<Scalar>;
      using Data = MultibodyFreeFwdDataTpl<Scalar>;

      using Manifold = proxnlp::MultibodyPhaseSpace<Scalar>;

      using ManifoldPtr = shared_ptr<Manifold>;
      ManifoldPtr space_;
      MatrixXs actuation_matrix_;

      const Manifold& space() const { return *space_; }
      int ntau() const { return space_->getModel().nv; }

      MultibodyFreeFwdDynamicsTpl(const ManifoldPtr& state, const MatrixXs& actuation);

      virtual void forward(const ConstVectorRef& x, const ConstVectorRef& u, ODEData& data) const;
      virtual void dForward(const ConstVectorRef& x, const ConstVectorRef& u, ODEData& data) const;

      shared_ptr<ContDataAbstract> createData() const;
    };
  
    template<typename Scalar>
    struct MultibodyFreeFwdDataTpl : ODEDataTpl<Scalar>
    {
      using VectorXs = typename math_types<Scalar>::VectorXs;
      using MatrixXs = typename math_types<Scalar>::MatrixXs;
      VectorXs tau_;
      MatrixXs dtau_dx_;
      MatrixXs dtau_du_;
      /// shared_ptr to the underlying pinocchio::DataTpl object.
      shared_ptr<pinocchio::DataTpl<Scalar>> pin_data_;
      MultibodyFreeFwdDataTpl(const MultibodyFreeFwdDynamicsTpl<Scalar>* cont_dyn);
    };

  } // namespace dynamics
} // namespace proxddp

#include "proxddp/modelling/dynamics/multibody-free-fwd.hxx"

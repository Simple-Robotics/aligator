#pragma once

#include "proxddp/core/function.hpp"

namespace proxddp
{
  /**
   * @brief   A simple function \f$f(u) = [u_{\min} - u; u - u_{\max}]\f$.
   * 
   * @details This function should be used along proxnlp::NegativeOrthant to create control bound
   *          constraints
   *          \f[
   *            -u_\min \leq u \leq u_\max.
   *          \f]
   */
  template<typename _Scalar>
  struct ControlBoxFunctionTpl : StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    using Base = StageFunctionTpl<Scalar>;
    using Data = FunctionDataTpl<Scalar>;

    VectorXs umin_, umax_;

    /// @brief Standard constructor: takes state space dimension and desired control bounds as vectors.
    ControlBoxFunctionTpl(const int ndx, const VectorXs umin, const VectorXs umax);

    /// @brief Constructor which takes control bounds as scalars cast into vectors of appropriate dimension.
    ControlBoxFunctionTpl(const int ndx, const int nu, const Scalar umin, const Scalar umax);

    void evaluate(const ConstVectorRef&,
                  const ConstVectorRef& u,
                  const ConstVectorRef&,
                  Data& data) const;
    
    /**
     * @copybrief Base::computeJacobians()
     * @details   This implementation does nothing: the values of the Jacobians are
     *            already set in createData().
     */
    void computeJacobians(const ConstVectorRef&,
                          const ConstVectorRef&,
                          const ConstVectorRef&,
                          Data& data) const;

    /// @copybrief Base::createData()
    /// @details   This override sets the appropriate values of the Jacobians.
    virtual shared_ptr<Data> createData() const;
  };
  
} // namespace proxddp

#include "proxddp/modelling/control-box-function.hxx"

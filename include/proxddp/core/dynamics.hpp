/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/function.hpp"


namespace proxddp
{
  template<typename _Scalar>
  using DynamicsDataTpl = FunctionDataTpl<_Scalar>;

  /** @brief    Dynamics model.
   * 
   *  A dynamics model is a function  \f$f(x,u,x')\f$ that must be set to zero,
   *  describing the dynamics mapping \f$(x, u) \mapsto x'\f$.
   */
  template<typename _Scalar>
  struct DynamicsModelTpl : StageFunctionTpl<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_FUNCTION_TYPEDEFS(Scalar)
    using Base = StageFunctionTpl<Scalar>;
    using Data = DynamicsDataTpl<Scalar>;

    /** @brief  Constructor for dynamics.
     * 
     * @param   ndx1 State space dimension for the current time node
     * @param   nu   Control dimension
     * @param   ndx2 Next state space dimension.
     */
    DynamicsModelTpl(const int ndx1,
                     const int nu,
                     const int ndx2)
      : StageFunctionTpl<Scalar>(ndx1, nu, ndx2, ndx2) {}

    /** @brief  @copybrief DynamicsModelTpl This constructor assumes same
     *          dimension for the current and next state.
     * 
     * @param   ndx State space dimension for the current and next time nodes
     * @param   nu   Control dimension
     */
    DynamicsModelTpl(const int ndx,
                     const int nu)
      : DynamicsModelTpl<Scalar>(ndx, nu, ndx) {}

  };


} // namespace proxddp

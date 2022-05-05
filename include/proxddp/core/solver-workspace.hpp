#pragma once
#include "proxddp/fwd.hpp"


namespace proxddp
{
  namespace internal
  {
    
    /// @brief  Contiguous storage for the value function parameters.
    ///
    /// @details This provides storage for the matrix \f[
    ///     \begin{bmatrix} 2v & V_x^\top \\ V_x & V_{xx} \end{bmatrix}
    /// \f]
    template<typename Scalar>
    struct value_storage
    {
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
      MatrixXs storage;
      Scalar& v_2_;
      VectorRef Vx_;
      MatrixRef Vxx_;

      value_storage(const int ndx)
        : storage(ndx + 1, ndx + 1)
        , v_2_(storage.coeffRef(0, 0))
        , Vx_(storage.bottomLeftCorner(ndx, 1))
        , Vxx_(storage.bottomRightCorner(ndx, ndx)) {
        storage.setZero();
      }
    };

    template<typename Scalar>
    struct q_function_storage
    {
      PROXNLP_DYNAMIC_TYPEDEFS(Scalar)
      const int ntot;

      MatrixXs storage;

      Scalar& q_2_; 

      VectorRef grad_;
      MatrixRef hess_;

      VectorRef Qx_;
      VectorRef Qu_;
      VectorRef Qy_;

      MatrixRef Qxx_;
      MatrixRef Qxu_;
      MatrixRef Qxy_;
      MatrixRef Quu_;
      MatrixRef Quy_;
      MatrixRef Qyy_;

      q_function_storage(const int ndx1, const int nu, const int ndx2)
        : ntot(ndx1 + nu + ndx2)
        , storage(ntot + 1, ntot + 1)
        , q_2_(storage.coeffRef(0, 0))
        , grad_(storage.bottomLeftCorner(ntot, 1))
        , hess_(storage.bottomRightCorner(ntot, ntot))
        , Qx_(grad_.head(ndx1))
        , Qu_(grad_.segment(ndx1, nu))
        , Qy_(grad_.tail(ndx2))
        , Qxx_(hess_.topLeftCorner(ndx1, ndx1))
        , Qxu_(hess_.block(0, ndx1, ndx1, nu))
        , Qxy_(hess_.topRightCorner(ndx1, ndx2))
        , Quu_(hess_.block(ndx1, ndx1, nu, nu))
        , Quy_(hess_.block(ndx1, ndx1 + nu, nu, ndx2))
        , Qyy_(hess_.bottomRightCorner(ndx2, ndx2)) {
        storage.setZero();
      }

    };

  } // namespace internal
} // namespace proxddp

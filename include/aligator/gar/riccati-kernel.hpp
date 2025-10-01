/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "aligator/context.hpp"
#include "lqr-problem.hpp"
#include "blk-matrix.hpp"
#include "aligator/core/bunchkaufman.hpp"

#include <boost/core/make_span.hpp>

#include <optional>

namespace aligator {
namespace gar {

/// Create a boost::span object from a vector and two indices.
template <class T, class A>
inline boost::span<T> make_span_from_indices(std::vector<T, A> &vec, size_t i0,
                                             size_t i1) {
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// @copybrief make_span_from_indices
template <class T, class A>
inline boost::span<const T> make_span_from_indices(const std::vector<T, A> &vec,
                                                   size_t i0, size_t i1) {
  return boost::make_span(vec.data() + i0, i1 - i0);
}

/// Per-node struct for all computations in the factorization.
template <typename _Scalar> struct StageFactor {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using allocator_type = ::aligator::polymorphic_allocator;

  struct CostToGo {
    using allocator_type = ::aligator::polymorphic_allocator;
    ArenaMatrix<MatrixXs> Vxx; //< "cost-to-go" matrix
    ArenaMatrix<VectorXs> vx;  //< "cost-to-go" gradient
    ArenaMatrix<MatrixXs> Vxt; //< cross-Hessian
    ArenaMatrix<MatrixXs> Vtt; //< parametric Hessian
    ArenaMatrix<VectorXs> vt;  //< parametric vector

    CostToGo(uint nx, uint nth, const allocator_type &alloc = {})
        : Vxx(nx, nx, alloc)
        , vx(nx, alloc)
        , Vxt(nx, nth, alloc)
        , Vtt(nth, nth, alloc)
        , vt(nth, alloc) {
      Vxt.setZero();
      Vtt.setZero();
      vt.setZero();
    }

    allocator_type get_allocator() const { return Vxx.get_allocator(); }

    CostToGo(const CostToGo &other, const allocator_type &alloc = {})
        : Vxx(other.Vxx, alloc)
        , vx(other.vx, alloc)
        , Vxt(other.Vxt, alloc)
        , Vtt(other.Vtt, alloc)
        , vt(other.vt, alloc) {}
    CostToGo(CostToGo &&other) noexcept = default;
    CostToGo(CostToGo &&other, const allocator_type &alloc)
        : Vxx(std::move(other.Vxx), alloc)
        , vx(std::move(other.vx), alloc)
        , Vxt(std::move(other.Vxt), alloc)
        , Vtt(std::move(other.Vtt), alloc)
        , vt(std::move(other.vt), alloc) {}
    CostToGo &operator=(const CostToGo &) = default;
    CostToGo &operator=(CostToGo &&) = default;
  };

  StageFactor(uint nx, uint nu, uint nc, uint nx2, uint nth,
              const allocator_type &alloc = {});

  allocator_type get_allocator() const { return Qhat.get_allocator(); }

  StageFactor(const StageFactor &other, const allocator_type &alloc = {});
  StageFactor(StageFactor &&) noexcept = default;
  StageFactor(StageFactor &&other, const allocator_type &alloc);

  StageFactor &operator=(const StageFactor &) = default;
  StageFactor &operator=(StageFactor &&) = default;
  ~StageFactor() = default;

  uint nx, nu, nc, nx2, nth;
  ArenaMatrix<MatrixXs> Qhat;
  ArenaMatrix<MatrixXs> Rhat;
  ArenaMatrix<MatrixXs> Shat;
  ArenaMatrix<VectorXs> qhat;
  ArenaMatrix<VectorXs> rhat;
  ArenaMatrix<RowMatrixXs> AtV;
  ArenaMatrix<RowMatrixXs> BtV;
  ArenaMatrix<MatrixXs> Gxhat;
  ArenaMatrix<MatrixXs> Guhat;
  BlkMatrix<VectorXs, 3, 1> ff;     //< feedforward gains
  BlkMatrix<RowMatrixXs, 3, 1> fb;  //< feedback gains
  BlkMatrix<RowMatrixXs, 3, 1> fth; //< parameter feedback gains
  BlkMatrix<MatrixXs, 2, 2> kktMat; //< reduced KKT matrix buffer
  BunchKaufman<MatrixXs> kktChol;   //< reduced KKT LDLT solver
  CostToGo vm;                      //< cost-to-go parameters
};

/// @brief Kernel for use in Riccati-like algorithms for the proximal LQ
/// subproblem.
template <typename Scalar> struct ProximalRiccatiKernel {
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using KnotType = LqrKnotTpl<Scalar>;
  using StageFactorType = StageFactor<Scalar>;
  using CostToGo = typename StageFactorType::CostToGo;

  struct kkt0_t {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> ff;
    BlkMatrix<RowMatrixXs, 2, 1> fth;
    BunchKaufman<MatrixXs> chol{mat.rows()};
    kkt0_t(uint nx, uint nc, uint nth)
        : mat({nx, nc}, {nx, nc})
        , ff(mat.rowDims())
        , fth(mat.rowDims(), {nth}) {}
  };

  static void terminalSolve(const KnotType &model, const Scalar mueq,
                            StageFactorType &d);

  static bool backwardImpl(boost::span<const KnotType> stages,
                           const Scalar mueq,
                           boost::span<StageFactorType> datas);

  /// Solve initial stage
  static void computeInitial(VectorRef x0, VectorRef lbd0, const kkt0_t &kkt0,
                             const std::optional<ConstVectorRef> &theta_);

  static void stageKernelSolve(const KnotType &model, StageFactorType &d,
                               CostToGo &vn, const Scalar mueq);

  /// Forward sweep.
  static bool
  forwardImpl(boost::span<const KnotType> stages,
              boost::span<const StageFactorType> datas,
              boost::span<VectorXs> xs, boost::span<VectorXs> us,
              boost::span<VectorXs> vs, boost::span<VectorXs> lbdas,
              const std::optional<ConstVectorRef> &theta_ = std::nullopt);
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct StageFactor<context::Scalar>;
extern template struct ProximalRiccatiKernel<context::Scalar>;
#endif

} // namespace gar
} // namespace aligator

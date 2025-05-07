/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
/// @author Wilson Jallet
#pragma once

#include "riccati-base.hpp"
#include "dense-kernel.hpp"

namespace aligator::gar {

/// @brief A stagewise-dense Riccati solver.
/// This algorithm uses a dense Bunch-Kaufman factorization at every stage.
/// @remark This is the approach from the T-RO journal submission and 2022 IROS
/// paper.
template <typename _Scalar>
class RiccatiSolverDense : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS_WITH_ROW_TYPES(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using BlkMat44 = BlkMatrix<MatrixXs, 4, 4>;
  using BlkRowMat41 = BlkMatrix<RowMatrixXs, 4, 1>;
  using BlkVec4 = BlkMatrix<VectorXs, 4, 1>;
  using KnotType = LqrKnotTpl<Scalar>;
  using Kernel = DenseKernel<Scalar>;
  using Data = typename Kernel::Data;

  std::vector<Data> stage_factors;
  std::vector<MatrixXs> Pxx;
  std::vector<MatrixXs> Pxt;
  std::vector<MatrixXs> Ptt;
  std::vector<VectorXs> px;
  std::vector<VectorXs> pt;
  struct {
    BlkMatrix<MatrixXs, 2, 2> mat;
    BlkMatrix<VectorXs, 2, 1> ff;
    BlkMatrix<MatrixXs, 2, 1> fth; // parametric rhs
    Eigen::BunchKaufman<MatrixXs> ldl;
  } kkt0;
  VectorXs thGrad;
  MatrixXs thHess;

  explicit RiccatiSolverDense(const LqrProblemTpl<Scalar> &problem);

  bool backward(const Scalar mudyn, const Scalar mueq);

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
               const std::optional<ConstVectorRef> &theta = std::nullopt) const;

  void cycleAppend(const KnotType &knot);
  VectorRef getFeedforward(size_t i) { return stage_factors[i].ff; }
  RowMatrixRef getFeedback(size_t i) { return stage_factors[i].fb; }

protected:
  const LqrProblemTpl<Scalar> *problem_;
};

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template class RiccatiSolverDense<context::Scalar>;
#endif
} // namespace aligator::gar

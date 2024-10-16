/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
/// @author Wilson Jallet
#pragma once

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>

#include "fwd.hpp"
#include "blk-matrix.hpp"
#include "riccati-base.hpp"

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
  using KnotType = LQRKnotTpl<Scalar>;

  std::vector<BlkMat44> kkts;
  std::vector<BlkVec4> ffs;
  std::vector<BlkRowMat41> fbs;
  std::vector<BlkRowMat41> fts;
  std::vector<Eigen::BunchKaufman<MatrixXs>> ldls;
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

  explicit RiccatiSolverDense(const LQRProblemTpl<Scalar> &problem);

  bool backward(const Scalar mudyn, const Scalar mueq);

  bool forward(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
               std::vector<VectorXs> &vs, std::vector<VectorXs> &lbdas,
               const std::optional<ConstVectorRef> &theta = std::nullopt) const;

  void cycleAppend(const KnotType &knot);
  VectorRef getFeedforward(size_t i) { return ffs[i].matrix(); }
  RowMatrixRef getFeedback(size_t i) { return fbs[i].matrix(); }

protected:
  void init_factor(const LQRKnotTpl<Scalar> &knot);
  void initialize();
  const LQRProblemTpl<Scalar> *problem_;
};

} // namespace aligator::gar

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "dense-riccati.txx"
#endif

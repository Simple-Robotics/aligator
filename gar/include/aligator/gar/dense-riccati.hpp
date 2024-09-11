#pragma once

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>

#include "blk-matrix.hpp"
#include "lqr-problem.hpp"
#include "riccati-base.hpp"
#include <aligator/tracy.hpp>

namespace aligator::gar {

/// A stagewise-dense Riccati solver
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

  struct FactorData {
    BlkMat44 kkt;
    BlkVec4 ff;
    BlkRowMat41 fb;
    BlkRowMat41 fth;
    Eigen::BunchKaufman<MatrixXs> ldl;
  };

  static FactorData init_factor(const LQRKnotTpl<Scalar> &knot) {
    std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
    using ldl_t = decltype(FactorData::ldl);
    long ntot = std::accumulate(dims.begin(), dims.end(), 0);
    uint nth = knot.nth;
    return FactorData{BlkMat44::Zero(dims, dims), BlkVec4::Zero(dims, {1}),
                      BlkRowMat41::Zero(dims, {knot.nx}),
                      BlkRowMat41::Zero(dims, {nth}), ldl_t{ntot}};
  }

  std::vector<FactorData> datas;
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

  VectorRef getFeedforward(size_t i) { return datas[i].ff.matrix(); }
  RowMatrixRef getFeedback(size_t i) { return datas[i].fb.matrix(); }

protected:
  void initialize();
  const LQRProblemTpl<Scalar> *problem_;
};

} // namespace aligator::gar

#include "dense-riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "dense-riccati.txx"
#endif

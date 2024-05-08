#pragma once

#include <proxsuite-nlp/linalg/bunchkaufman.hpp>

#include "blk-matrix.hpp"
#include "lqr-problem.hpp"
#include "riccati-base.hpp"
#include <tracy/Tracy.hpp>

namespace aligator::gar {

/// A stagewise-dense Riccati solver
template <typename _Scalar>
class RiccatiSolverDense : public RiccatiSolverBase<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = RiccatiSolverBase<Scalar>;
  using BlkMat44 = BlkMatrix<MatrixXs, 4, 4>;
  using BlkMat41 = BlkMatrix<MatrixXs, 4, 1>;
  using BlkVec4 = BlkMatrix<VectorXs, 4, 1>;
  using KnotType = LQRKnotTpl<Scalar>;

  struct FactorData {
    BlkMat44 kkt;
    BlkVec4 ff;
    BlkMat41 fb;
    BlkMat41 fth;
    Eigen::BunchKaufman<MatrixXs> ldl;
  };

  static FactorData init_factor(const LQRKnotTpl<Scalar> &knot) {
    std::array<long, 4> dims = {knot.nu, knot.nc, knot.nx2, knot.nx2};
    using ldl_t = decltype(FactorData::ldl);
    long ntot = std::accumulate(dims.begin(), dims.end(), 0);
    uint nth = knot.nth;
    return FactorData{BlkMat44(dims, dims), BlkVec4(dims, {1}),
                      BlkMat41(dims, {knot.nx}), BlkMat41(dims, {nth}),
                      ldl_t{ntot}};
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

protected:
  void initialize();
  const LQRProblemTpl<Scalar> *problem_;
};

} // namespace aligator::gar

#include "dense-riccati.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "dense-riccati.txx"
#endif

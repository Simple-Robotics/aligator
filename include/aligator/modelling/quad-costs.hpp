#pragma once

#include "proxddp/core/cost-abstract.hpp"
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace aligator {

template <typename Scalar> struct QuadraticCostDataTpl;

/// @brief Euclidean quadratic cost.
template <typename _Scalar> struct QuadraticCostTpl : CostAbstractTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using Data = QuadraticCostDataTpl<Scalar>;
  using VectorSpace = proxsuite::nlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  /// Weight @f$ Q @f$
  MatrixXs weights_x;
  /// Weight @f$ R @f$
  MatrixXs weights_u;

protected:
  /// Weight N for term @f$ x^\top N u @f$
  MatrixXs weights_cross_;

  static void check_dim_equal(long n, long m, const std::string &msg = "") {
    if (n != m)
      ALIGATOR_RUNTIME_ERROR(fmt::format(
          "Dimensions inconsistent: got {:d} and {:d}{}.\n", n, m, msg));
  }

  void debug_check_dims() const {
    check_dim_equal(weights_x.rows(), weights_x.cols(), " for x weights");
    check_dim_equal(weights_u.rows(), weights_u.cols(), " for u weights");
    check_dim_equal(weights_cross_.rows(), this->ndx(),
                    " for cross-term weight");
    check_dim_equal(weights_cross_.cols(), this->nu, " for cross-term weight");
    check_dim_equal(interp_x.rows(), weights_x.rows(),
                    " for x weights and intercept");
    check_dim_equal(interp_u.rows(), weights_u.rows(),
                    " for u weights and intercept");
  }

public:
  VectorXs interp_x;
  VectorXs interp_u;

  static auto get_vector_space(Eigen::Index nx) {
    return std::make_shared<VectorSpace>((int)nx);
  }

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u,
                   const ConstVectorRef &interp_x,
                   const ConstVectorRef &interp_u)
      : Base(get_vector_space(w_x.cols()), (int)w_u.cols()), weights_x(w_x),
        weights_u(w_u), weights_cross_(this->ndx(), this->nu),
        interp_x(interp_x), interp_u(interp_u), has_cross_term_(false) {
    debug_check_dims();
    weights_cross_.setZero();
  }

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u,
                   const ConstMatrixRef &w_cross,
                   const ConstVectorRef &interp_x,
                   const ConstVectorRef &interp_u)
      : Base(get_vector_space(w_x.cols()), (int)w_u.cols()), weights_x(w_x),
        weights_u(w_u), weights_cross_(w_cross), interp_x(interp_x),
        interp_u(interp_u), has_cross_term_(true) {
    debug_check_dims();
  }

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u)
      : QuadraticCostTpl(w_x, w_u, VectorXs::Zero(w_x.cols()),
                         VectorXs::Zero(w_u.cols())) {}

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u,
                   const ConstMatrixRef &w_cross)
      : QuadraticCostTpl(w_x, w_u, w_cross, VectorXs::Zero(w_x.cols()),
                         VectorXs::Zero(w_u.cols())) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const {
    Data &d = static_cast<Data &>(data);
    d.w_times_x_.noalias() = weights_x * x;
    d.w_times_u_.noalias() = weights_u * u;
    if (has_cross_term_) {
      d.cross_x_.noalias() = weights_cross_ * u;
      d.cross_u_.noalias() = weights_cross_.transpose() * x;

      d.w_times_x_ += d.cross_x_;
      d.w_times_u_ += d.cross_u_;
    }
    data.value_ = Scalar(0.5) * x.dot(d.w_times_x_ + 2 * interp_x) +
                  Scalar(0.5) * u.dot(d.w_times_u_ + 2 * interp_u);
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &data) const {
    Data &d = static_cast<Data &>(data);
    d.Lx_ = d.w_times_x_ + interp_x;
    d.Lu_ = d.w_times_u_ + interp_u;
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const {}

  shared_ptr<CostData> createData() const {
    auto data = std::make_shared<Data>(this->ndx(), this->nu);
    data->Lxx_ = weights_x;
    data->Luu_ = weights_u;
    data->Lxu_ = weights_cross_;
    data->Lux_ = weights_cross_.transpose();
    return data;
  }

  const ConstMatrixRef getCrossWeights() const { return weights_cross_; }
  void setCrossWeight(const ConstMatrixRef &w) {
    weights_cross_ = w;
    has_cross_term_ = true;
    debug_check_dims();
  }

  /// @copydoc has_cross_term_
  bool hasCrossTerm() const { return has_cross_term_; }

protected:
  /// Whether a cross term exists
  bool has_cross_term_;
};

template <typename Scalar>
struct QuadraticCostDataTpl : CostDataAbstractTpl<Scalar> {
  using Base = CostDataAbstractTpl<Scalar>;
  using VectorXs = typename Base::VectorXs;
  VectorXs w_times_x_, w_times_u_, cross_x_, cross_u_;

  QuadraticCostDataTpl(const int nx, const int nu)
      : Base(nx, nu), w_times_x_(nx), w_times_u_(nu), cross_x_(nu),
        cross_u_(nu) {
    w_times_x_.setZero();
    w_times_u_.setZero();
    cross_x_.setZero();
    cross_u_.setZero();
  }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/quad-costs.txx"
#endif

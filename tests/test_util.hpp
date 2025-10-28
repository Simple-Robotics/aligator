#include <Eigen/Core>
#include <random>

struct normal_unary_op {
  mutable std::mt19937 rng;
  mutable std::normal_distribution<double> gen;

  explicit normal_unary_op(std::mt19937 rng, double stddev = 1.0)
      : rng(rng)
      , gen(0.0, stddev) {}

  void set_seed(size_t sd) { rng.seed(sd); }

  double operator()() const { return gen(rng); }
};

/// Generate a Wishart-distributed matrix in @p n dimensions with @p p DoF
inline Eigen::MatrixXd sampleWishartDistributedMatrix(uint n, uint p) {
  using Eigen::MatrixXd;
  std::mt19937 rng;
  MatrixXd root = MatrixXd::NullaryExpr(n, p, normal_unary_op(rng));
  return root * root.transpose();
}

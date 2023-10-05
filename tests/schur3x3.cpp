#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#include <chrono>
#include "aligator/context.hpp"
#include "aligator/threads.hpp"
#include <Eigen/Cholesky>

using namespace aligator::context;
using namespace std::chrono;
namespace utf = boost::unit_test;

constexpr double EPSTOL = 1e-9;

/// Represents a structured 3x3 block-banded matrix of the form
/// \f[ \begin{bmatrix}
/// G & J^\top & \\ J & \Lambda & U \\ & U^\top & H
/// \end{bmatrix} \f]
struct Band3x3Matrix {
  uint nx;
  uint ny;
  uint nz;
  uint ntot = nx + ny + nz;

  MatrixXs data;

  MatrixRef G; // UpLeft
  MatrixRef H; // LoRight
  MatrixRef Lambda;
  MatrixRef J;
  MatrixRef U;

  Band3x3Matrix(uint nx, uint ny, uint nz)
      : nx(nx), ny(ny), nz(nz), data(ntot, ntot), //
        G(data.block(0, 0, nx, nx)), H(data.block(nx + ny, nx + ny, nz, nz)),
        Lambda(data.block(nx, nx, ny, ny)), J(data.block(nx, 0, ny, nx)),
        U(data.block(nx, nx + ny, ny, nz)) {
    data.setZero();
  }

  MatrixXs toDense() const {
    auto out = data;
    out.topRows(nx).middleCols(nx, ny) = J.transpose();
    out.bottomRows(nz).middleCols(nx, ny) = U.transpose();
    return out;
  }
};

constexpr int kNumThreads = 2;

/// Solve \f$ M [x;y;z] = [b;c;d]\f$ in-place.
template <typename RhsType>
auto structuredSolve(const Band3x3Matrix &M, RhsType &rhsAndX) {
  using namespace aligator::omp;
  // 1. split the rhs
  auto x = rhsAndX.topRows(M.nx);
  auto y = rhsAndX.middleRows(M.nx, M.ny);
  auto z = rhsAndX.bottomRows(M.nz);

  // 2. solve each subsystem
  // allocate cholesky things
  Eigen::LLT<MatrixXs> cholG(M.G);
  Eigen::LLT<MatrixXs> cholH(M.H);

  // solution buffers
  MatrixXs Kx = -M.J.transpose();
  MatrixXs Kz = -M.U.transpose();

#pragma omp parallel sections
  {
#pragma omp section
    {
      cholG.solveInPlace(x);
      cholG.solveInPlace(Kx);
    }
#pragma omp section
    {
      cholH.solveInPlace(z);
      cholH.solveInPlace(Kz);
    }
  }

  MatrixXs Yschur = M.Lambda;
  // add gain terms
  Yschur.noalias() += M.J * Kx;
  Yschur.noalias() += M.U * Kz;

  y.noalias() -= M.J * x;
  y.noalias() -= M.U * z;

  auto cholYs = Yschur.ldlt();
  cholYs.solveInPlace(y);

  x.noalias() += Kx * y;
  z.noalias() += Kz * y;
}

auto diagSmallProbLhs() {
  uint nx = 1;
  uint ny = 2;
  uint nz = 1;
  Band3x3Matrix M(nx, ny, nz);
  M.G.setRandom();
  M.G.noalias() = M.G.transpose() * M.G;
  M.H.setRandom();
  M.H.noalias() = M.H.transpose() * M.H;

  auto D(M.Lambda.diagonal());
  D.setRandom();
  D = D * D;
  return M;
}

auto fullProbLhs(uint nz) {
  uint nx = nz;
  uint ny = 2;
  Band3x3Matrix M(nx, ny, nz);
  M.G.setIdentity();
  M.G *= 0.4;
  MatrixXs Hf(nz, nz);
  Hf.setRandom();
  M.H.noalias() = Hf.transpose() * Hf;

  auto D = M.Lambda.diagonal();
  D.setRandom();
  D = D * D;
  D.array() += 1e-4;
  M.J.setOnes();
  M.U.setRandom();
  return M;
}

std::tuple<long, long> runTest(const Band3x3Matrix M, bool print = false) {
  auto view = M.toDense();

  if (print) {
    fmt::print("dense matrix:\n{}\n", view);
  }

  VectorXs rhs1(M.ntot);
  rhs1.setRandom();

  VectorXs sol1 = rhs1;
  auto start = high_resolution_clock::now();
  structuredSolve(M, sol1);
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  long dur2;

  {
    auto start = high_resolution_clock::now();
    VectorXs sol2 = view.ldlt().solve(rhs1);
    auto stop = high_resolution_clock::now();
    dur2 = duration_cast<microseconds>(stop - start).count();

    BOOST_CHECK(sol1.isApprox(sol2, EPSTOL));
  }
  return {duration.count(), dur2};
}

template <typename T> auto getAvg(std::vector<T> const &data) {
  if (data.empty())
    return 0.0;
  return std::accumulate(data.begin(), data.end(), 0.0) / (T)data.size();
}

constexpr int NUM_TIMINGS = 500;

BOOST_AUTO_TEST_CASE(band3x3_diag) {
  std::vector<double> times1, times2;
  for (int i = 0; i < NUM_TIMINGS; i++) {
    double t1, t2;
    std::tie(t1, t2) = runTest(diagSmallProbLhs());
    times1.push_back(t1);
    times2.push_back(t2);
  }
  fmt::print("Avg time (µs) (structured): {}\n", getAvg(times1));
  fmt::print("Avg time (µs) (Eigen LDLT): {}\n", getAvg(times2));
  fmt::print("----\n");
}

BOOST_AUTO_TEST_CASE(band3x3_full) {
  for (int n : {4, 12, 32, 64, 128}) {
    std::vector<double> times1, times2;
    for (int i = 0; i < NUM_TIMINGS; i++) {
      double t1, t2;
      std::tie(t1, t2) = runTest(fullProbLhs((unsigned)n));
      times1.push_back(t1);
      times2.push_back(t2);
    }
    fmt::print("Avg time (µs) (structured): {}\n", getAvg(times1));
    fmt::print("Avg time (µs) (Eigen LDLT): {}\n", getAvg(times2));
    fmt::print("----\n");
  }
}

int main(int argc, char **argv) {
  printf("NUMBER OF AVAILABLE THREADS: %lu\n",
         aligator::omp::get_available_threads());
  aligator::omp::set_default_options(kNumThreads);

  int ret = utf::unit_test_main(&init_unit_test, argc, argv);

  return ret;
}

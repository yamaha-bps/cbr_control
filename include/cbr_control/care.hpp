// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__CARE_HPP_
#define CBR_CONTROL__CARE_HPP_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <type_traits>
#include <numeric>

namespace cbr
{

/* -------------------------------------------------------------------------- */
/*                               Matrix Balance                               */
/* -------------------------------------------------------------------------- */

// Numerical Recipes p.593 - Balancing Trasformation
// D^-1 * A * D = B

template<std::size_t nx, class T = double>
void matrix_balance(
  const Eigen::Ref<const Eigen::Matrix<T, 2 * nx, 2 * nx>> A,
  Eigen::Ref<Eigen::Matrix<T, 2 * nx, 2 * nx>> D,
  Eigen::Ref<Eigen::Matrix<T, 2 * nx, 2 * nx>> B
)
{
  std::size_t n = 2 * nx;

  // Initialize D and B
  D.setIdentity();
  B = A;

  double RADIX = 2.0;
  double sqrdx = RADIX * RADIX;

  std::size_t done = 0;

  while (done != 1) {
    done = 1;

    for (std::size_t i = 0; i < n; i++) {
      double r = 0.;
      double c = 0.;

      for (std::size_t j = 0; j < n; j++) {
        if (j != i) {
          c = c + std::abs(B(j, i));
          r = r + std::abs(B(i, j));
        }
      }

      if ((c != 0) && (r != 0)) {
        double g = r / RADIX;
        double f = 1.0;
        double s = c + r;

        while (c < g) {
          f = f * RADIX;
          c = c * sqrdx;
        }

        g = r * RADIX;

        while (c > g) {
          f = f / RADIX;
          c = c / sqrdx;
        }

        if (((c + r) / f) < 0.95 * s) {
          done = 0;
          g = 1 / f;
          D(i, i) = D(i, i) * f;

          for (std::size_t j = 0; j < n; j++) {
            B(i, j) = B(i, j) / f;
          }

          for (std::size_t j = 0; j < n; j++) {
            B(j, i) = B(j, i) * f;
          }
        }
      }
    }
  }
}


// Finds the solution to the ARE by finding the Eigen-decomposition of the Hamiltonean.
template<std::size_t nx, std::size_t nu, class T = double>
bool care(
  const Eigen::Ref<const Eigen::Matrix<T, nx, nx>> A,
  const Eigen::Ref<const Eigen::Matrix<T, nx, nu>> B,
  const Eigen::Ref<const Eigen::Matrix<T, nx, nx>> Q,
  const Eigen::Ref<const Eigen::Matrix<T, nu, nu>> R,
  Eigen::Ref<Eigen::Matrix<T, nx, nx>> P,
  Eigen::Ref<Eigen::Matrix<T, nu, nx>> K
)
{
// Ensure R positive definite (R>0)
  const Eigen::LLT<Eigen::Matrix<T, nu, nu>> Rdecomposed(R.transpose());
  if (Rdecomposed.info() == Eigen::NumericalIssue) {
    return false;
  }

  using H_t = Eigen::Matrix<T, 2 * nx, 2 * nx>;
  using Hc_t = Eigen::Matrix<std::complex<T>, 2 * nx, 2 * nx>;
  using Ac_t = Eigen::Matrix<std::complex<T>, nx, nx>;

//  1. Define Hamiltonean:
//
//                      [  A | -(B/R)*B' ]
//                H =   [ ---|-----------]
//                      [ -Q |    -A'    ]

  H_t H;
  H.template topLeftCorner<nx, nx>() = A;
  H.template topRightCorner<nx,
    nx>() = -Rdecomposed.solve(B.transpose()).transpose() * B.transpose();
  H.template bottomLeftCorner<nx, nx>() = -Q;
  H.template bottomRightCorner<nx, nx>() = -A.transpose();

  //  2. Balance the Hamiltonean
  H_t D;
  H_t Hb;
  matrix_balance<nx, T>(H, D, Hb);

  //  3. Solve the ARE through eigen decomposition of Hb
  // Start by obtaining eigenvalues and eigenvectors
  const Eigen::EigenSolver<H_t> es(Hb);
  auto V = es.eigenvectors();
  const auto & l = es.eigenvalues();
  V = D * V;

  // Idendify which eigenvalues are positive and which are negative
  std::array<int, 2 * nx> ord_L;
  std::fill(ord_L.begin(), ord_L.end(), 0);

  for (std::size_t k = 0; k < (2 * nx); k++) {
    if (std::real(l[k]) < 0) {
      ord_L[k] = -1;
    } else if (std::real(l[k]) > 0) {
      ord_L[k] = 1;
    }
  }

  // Sorting indices - place positive on left side, negative on right
  std::array<std::size_t, 2 * nx> ord_index;
  std::iota(ord_index.begin(), ord_index.end(), 0LU);
  std::sort(
    ord_index.begin(), ord_index.end(),
    [&](const std::size_t i1, const std::size_t i2) {
      const auto & l1 = ord_L[i1];
      const auto & l2 = ord_L[i2];
      return l1 > l2;
    });

  // Sort Eigenvector based on ord_index array
  Hc_t V_ord;
  for (std::size_t i = 0; i < 2 * nx; i++) {
    V_ord.col(i) = V.col(ord_index[i]);
  }

  // Define upper and lower right side block matrices
  const Ac_t V12 = V_ord.template topRightCorner<nx, nx>();
  const Ac_t V22 = V_ord.template bottomRightCorner<nx, nx>();

  const Eigen::FullPivLU<Ac_t> V12decomposed(V12.transpose());
  const Ac_t P_complex = V12decomposed.solve(V22.transpose()).transpose();  // P_complex = T22/T12

  // 4. Write Results
  P = P_complex.unaryExpr([](const std::complex<T> & v) {return std::real(v);});

  K = R.inverse() * B.transpose() * P;

  return true;
}

}    // namespace cbr


#endif  // CBR_CONTROL__CARE_HPP_

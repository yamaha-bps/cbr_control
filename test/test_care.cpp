// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <gtest/gtest.h>
#include <cbr_control/care.hpp>

template void cbr::matrix_balance<4, double>(
  const Eigen::Ref<const Eigen::Matrix<double, 8, 8>> A,
  Eigen::Ref<Eigen::Matrix<double, 8, 8>> D,
  Eigen::Ref<Eigen::Matrix<double, 8, 8>> B
);

template bool cbr::care<4, 1, double>(
  const Eigen::Ref<const Eigen::Matrix<double, 4, 4>> A,
  const Eigen::Ref<const Eigen::Matrix<double, 4, 1>> B,
  const Eigen::Ref<const Eigen::Matrix<double, 4, 4>> Q,
  const Eigen::Ref<const Eigen::Matrix<double, 1, 1>> R,
  Eigen::Ref<Eigen::Matrix<double, 4, 4>> P,
  Eigen::Ref<Eigen::Matrix<double, 1, 4>> K
);

TEST(Care, Results)
{
  constexpr double eps = 1e-8;

  const auto nx = 4;
  const auto nu = 1;

  // Define Matrices
  Eigen::Matrix<double, nx, nx> A;
  Eigen::Matrix<double, nx, nu> B;
  Eigen::Matrix<double, nx, nx> Q;
  Eigen::Matrix<double, nu, nu> R;

  Eigen::Matrix<double, nx, nx> P;
  Eigen::Matrix<double, nu, nx> K;

  Eigen::Matrix<double, nx, nx> P_ref;
  Eigen::Matrix<double, nu, nx> K_ref;

  // Define Known Problem:
  // Inverted Pendulum example
  // https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace

  A <<
    0., 1.0000, 0., 0.,
    0., -0.181818181818182, 2.67272727272727, 0.,
    0., 0., 0., 1.0000,
    0, -0.454545454545455, 31.1818181818182, 0.;

  B <<
    0.,
    1.818181818181818,
    0.,
    4.545454545454545;

  Q <<
    1., 0., 0., 0.,
    0., 0., 0., 0.,
    0., 0., 1., 0.,
    0., 0., 0., 0.;

  R << 1.;

  // Using matlab, [K_ref, P_ref, ~] = lqr(A,B,Q,R)
  // We get the following reference K and P matrices.

  K_ref << -1.00000000000002, -1.65671002515709, 18.6853959018802, 3.45943817580003;

  P_ref <<
    1.55671002515708, 1.20667305121227, -3.45943817580007, -0.702669220484911,
    1.20667305121227, 1.45543614041921, -4.68267286919413, -0.946650661702243,
    -3.45943817580005, -4.68267286919410, 31.6320495422315, 5.98385624609128,
    -0.702669220484914, -0.946650661702243, 5.98385624609131, 1.13973666335690;


  bool result = cbr::care<nx, nu, double>(A, B, Q, R, P, K);

  if (result == true) {
    std::cout << "True" << std::endl;
  } else {
    std::cout << "False" << std::endl;
  }

  std::cout << "\nTest Results\n" << std::endl;
  std::cout << "P\n" << P << std::endl;
  std::cout << "Pref\n" << P_ref << std::endl;

  std::cout << "K\n" << K << std::endl;
  std::cout << "Kref\n" << K_ref << std::endl;


  for (std::size_t i = 0; i < nx * nu; i++) {
    ASSERT_NEAR(*(K.data() + i), *(K_ref.data() + i), eps);
  }

  for (std::size_t i = 0; i < nx * nx; i++) {
    ASSERT_NEAR(*(P.data() + i), *(P_ref.data() + i), eps);
  }
}

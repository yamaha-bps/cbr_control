// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef PLANARQUAD_DYNAMICS_HPP_
#define PLANARQUAD_DYNAMICS_HPP_

#include <Eigen/Dense>

// template here over some type
template<typename T1, typename T2>
auto planarquad_dynamics(const Eigen::MatrixBase<T1> & x, const Eigen::MatrixBase<T2> & u)
{
  // Define nx, nu
  constexpr auto nx = T1::RowsAtCompileTime;
  constexpr auto nu = T2::RowsAtCompileTime;

  // Initialize Parameters
  constexpr double I = 0.2;
  constexpr double m = 0.5;
  constexpr double r = 0.25;
  constexpr double gr = 9.81;

  using T = typename decltype(x * u.transpose())::EvalReturnType::Scalar;

  // xDot = f(x) + g(x)*u
  Eigen::Matrix<T, nx, 1> xDot;
  Eigen::Matrix<typename T1::Scalar, nx, 1> f;

  Eigen::Matrix<typename T1::Scalar, nx, nu> g;
  Eigen::Map<Eigen::Matrix<typename T1::Scalar, nx, 1>> g1(g.data());
  Eigen::Map<Eigen::Matrix<typename T1::Scalar, nx, 1>> g2(g.data() + nx);

  // Define States
  // const auto & X = x[0];
  // const auto & Y = x[1];
  const auto & theta = x[2];

  // Dynamics
  f[0] = x[3];
  f[1] = x[4];
  f[2] = x[5];
  f[3] = 0.;
  f[4] = -gr;
  f[5] = 0.;


  g1[0] = 0;                g2[0] = 0;
  g1[1] = 0;                g2[1] = 0;
  g1[2] = 0;                g2[2] = 0;
  g1[3] = -sin(theta) / m;  g2[3] = -sin(theta) / m;
  g1[4] = cos(theta) / m;   g2[4] = cos(theta) / m;
  g1[5] = r / I;            g2[5] = -r / I;

  xDot = f + g * u;

  return xDot;
}

#endif  // PLANARQUAD_DYNAMICS_HPP_

// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef SEGWAY_DYNAMICS_HPP_
#define SEGWAY_DYNAMICS_HPP_

#include <Eigen/Dense>

// template here over some type
template<typename T1, typename T2>
auto segway_dynamics(const Eigen::MatrixBase<T1> & x, const Eigen::MatrixBase<T2> & u)
{
  // Define nx, nu
  constexpr auto nx = T1::RowsAtCompileTime;
  constexpr auto nu = T2::RowsAtCompileTime;

  // Initialize Parameters
  constexpr double mb = 44.798;
  constexpr double mw = 2.485;
  constexpr double Jw = 0.055936595310797;
  constexpr double a2 = -0.02322718759275;
  constexpr double c2 = 0.166845864363019;
  constexpr double A2 = 3.604960049044268;
  constexpr double B2 = 3.836289730154863;
  constexpr double C2 = 1.069672194414735;
  constexpr double K = 1.261650363363571;
  constexpr double r = 0.195;
  constexpr double L = 0.5;
  constexpr double gGravity = 9.81;
  constexpr double FricCoeffViscous = 0.;
  constexpr double velEps = 1.0e-3;
  constexpr double FricCoeff = 1.225479467549329;

  using T = typename decltype(x * u.transpose())::EvalReturnType::Scalar;

  // xDot = f(x) + g(x)*u
  Eigen::Matrix<T, nx, 1> xDot;
  Eigen::Matrix<typename T1::Scalar, nx, 1> f;

  Eigen::Matrix<typename T1::Scalar, nx, nu> g;
  Eigen::Map<Eigen::Matrix<typename T1::Scalar, nx, 1>> g1(g.data());
  Eigen::Map<Eigen::Matrix<typename T1::Scalar, nx, 1>> g2(g.data() + nx);


  // Extract States
  const auto & theta = x[2];
  const auto & v = x[3];
  const auto & thetaDot = x[4];
  const auto & psi = x[5];
  const auto & psiDot = x[6];

  const auto Fric = FricCoeff * tanh((v - psiDot * r) / velEps) + FricCoeffViscous *
    (v - psiDot * r);

  f[0] = v * cos(theta);
  f[1] = v * sin(theta);
  f[2] = thetaDot;
  f[3] = (1 / 2) * r *
    (1 /
    (4 * B2 * Jw + 4 *
    pow(
      a2,
      2) * Jw * mb + 4 *
    pow(
      c2,
      2) * Jw * mb + 2 * B2 * mb *
    pow(
      r,
      2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) * pow(mb, 2) * pow(r, 2) + 4 * B2 * mw * pow(r, 2) + 4 * pow(a2, 2) * mb * mw * pow(r, 2) +
    4 *
    pow(
      c2,
      2) * mb * mw *
    pow(
      r,
      2) +
    (pow(
      a2,
      2) + (-1) *
    pow(
      c2,
      2)) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) * cos(2 * psi) + 2 * a2 * c2 *
    pow(mb, 2) * pow(r, 2) * sin(2 * psi))) * ((-8) * B2 * Fric + (-8) * pow(a2, 2) * Fric * mb +
    (-8) *
    pow(
      c2,
      2) * Fric * mb + mb * r *
    ((-8) * c2 * Fric + a2 * ((-1) * A2 + C2) *
    pow(thetaDot, 2) + 4 * a2 * B2 * (pow(psiDot, 2) + pow(thetaDot, 2)) +
    pow(
      a2,
      3) * mb *
    (4 *
    pow(
      psiDot,
      2) + 3 *
    pow(
      thetaDot,
      2)) + a2 *
    pow(
      c2,
      2) * mb *
    (4 *
    pow(
      psiDot,
      2) + 3 *
    pow(
      thetaDot,
      2))) * cos(psi) + (-4) * a2 * c2 * gGravity *
    pow(mb, 2) * r * cos(2 * psi) + a2 * A2 * mb * r * pow(thetaDot, 2) * cos(3 * psi) +
    (-1) * a2 * C2 * mb * r *
    pow(
      thetaDot,
      2) * cos(3 * psi) +
    pow(
      a2,
      3) *
    pow(
      mb,
      2) * r *
    pow(
      thetaDot,
      2) * cos(3 * psi) + (-3) * a2 *
    pow(c2, 2) * pow(mb, 2) * r * pow(thetaDot, 2) * cos(3 * psi) + 8 * a2 * Fric * mb * r * sin(
      psi) + 4 * B2 * c2 * mb * pow(psiDot, 2) * r * sin(psi) +
    4 *
    pow(
      a2,
      2) * c2 *
    pow(
      mb,
      2) *
    pow(
      psiDot,
      2) * r * sin(psi) + 4 *
    pow(
      c2,
      3) *
    pow(
      mb,
      2) *
    pow(
      psiDot,
      2) * r * sin(psi) + A2 * c2 * mb * r *
    pow(
      thetaDot,
      2) * sin(psi) + 4 * B2 * c2 * mb * r *
    pow(thetaDot, 2) * sin(psi) + (-1) * c2 * C2 * mb * r * pow(thetaDot, 2) * sin(psi) +
    3 *
    pow(
      a2,
      2) * c2 *
    pow(mb, 2) * r * pow(thetaDot, 2) * sin(psi) + 3 * pow(c2, 3) * pow(mb, 2) * r * pow(
      thetaDot,
      2) *
    sin(psi) + 2 *
    pow(a2, 2) * gGravity * pow(mb, 2) * r * sin(2 * psi) + (-2) * pow(c2, 2) * gGravity * pow(
      mb,
      2) *
    r * sin(2 * psi) + A2 * c2 * mb * r * pow(thetaDot, 2) * sin(3 * psi) +
    (-1) * c2 * C2 * mb * r *
    pow(
      thetaDot,
      2) * sin(3 * psi) + 3 *
    pow(
      a2,
      2) * c2 *
    pow(
      mb,
      2) * r *
    pow(
      thetaDot,
      2) * sin(3 * psi) + (-1) * pow(c2, 3) * pow(mb, 2) * r * pow(thetaDot, 2) * sin(3 * psi));
  f[4] =
    pow(
    r,
    2) * thetaDot *
    ((-2) * a2 * mb * v * cos(psi) + (-4) * a2 * c2 * mb * psiDot * cos(2 * psi) + (-2) *
    (c2 * mb * v +
    (A2 + (-1) * C2 + (-2) *
    pow(
      a2,
      2) * mb + 2 *
    pow(
      c2,
      2) * mb) * psiDot * cos(psi)) * sin(psi)) *
    (1 / (Jw * pow(L, 2) + pow(L, 2) * mw * pow(r, 2) +
    2 *
    (C2 +
    pow(
      a2,
      2) * mb) *
    pow(
      r,
      2) *
    pow(
      cos(psi),
      2) + 2 *
    (A2 +
    pow(c2, 2) * mb) * pow(r, 2) * pow(sin(psi), 2) + 2 * a2 * c2 * mb * pow(r, 2) * sin(2 * psi)));
  f[5] = psiDot;
  f[6] =
    (1 /
    (4 * B2 * Jw + 4 *
    pow(
      a2,
      2) * Jw * mb + 4 *
    pow(
      c2,
      2) * Jw * mb + 2 * B2 * mb *
    pow(
      r,
      2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) + 4 * B2 * mw *
    pow(
      r,
      2) + 4 *
    pow(
      a2,
      2) * mb * mw *
    pow(
      r,
      2) + 4 *
    pow(
      c2,
      2) * mb * mw *
    pow(r, 2) + (pow(a2, 2) + (-1) * pow(c2, 2)) * pow(mb, 2) * pow(r, 2) * cos(2 * psi) +
    2 * a2 * c2 *
    pow(
      mb,
      2) *
    pow(
      r,
      2) * sin(2 * psi))) *
    (8 * Fric * Jw + 4 * Fric * mb *
    pow(
      r,
      2) + 8 * Fric * mw *
    pow(
      r,
      2) + 2 * mb *
    (2 * c2 * Fric * r + a2 * gGravity *
    (2 * Jw + (mb + 2 * mw) *
    pow(r, 2))) * cos(psi) + (-2) * a2 * c2 * mb * (mb * pow(psiDot, 2) * pow(r, 2) +
    (-2) *
    (Jw + mw * pow(r, 2)) * pow(thetaDot, 2)) * cos(2 * psi) + 4 * c2 * gGravity * Jw * mb * sin(
      psi) + (-4) * a2 * Fric * mb * r * sin(psi) + 2 * c2 * gGravity *
    pow(mb, 2) * pow(r, 2) * sin(psi) + 4 * c2 * gGravity * mb * mw * pow(r, 2) * sin(psi) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      psiDot,
      2) *
    pow(
      r,
      2) * sin(2 * psi) + (-1) *
    pow(
      c2,
      2) *
    pow(
      mb,
      2) * pow(psiDot, 2) * pow(r, 2) * sin(2 * psi) + (-2) * A2 * Jw * pow(thetaDot, 2) * sin(
      2 * psi) + 2 * C2 * Jw *
    pow(
      thetaDot,
      2) * sin(2 * psi) + (-2) * pow(a2, 2) * Jw * mb * pow(thetaDot, 2) * sin(2 * psi) +
    2 *
    pow(
      c2,
      2) * Jw * mb *
    pow(
      thetaDot,
      2) * sin(2 * psi) + (-1) * A2 * mb *
    pow(
      r,
      2) *
    pow(
      thetaDot,
      2) * sin(2 * psi) + C2 * mb *
    pow(
      r,
      2) *
    pow(
      thetaDot,
      2) * sin(2 * psi) + (-2) * A2 * mw *
    pow(
      r,
      2) *
    pow(thetaDot, 2) * sin(2 * psi) + 2 * C2 * mw * pow(r, 2) * pow(thetaDot, 2) * sin(2 * psi) +
    (-2) *
    pow(
      a2,
      2) * mb * mw * pow(r, 2) * pow(thetaDot, 2) * sin(2 * psi) + 2 * pow(c2, 2) * mb * mw * pow(
      r,
      2) *
    pow(thetaDot, 2) * sin(2 * psi));

  g1[0] = 0;
  g2[0] = 0;

  g1[1] = 0;
  g2[1] = 0;

  g1[2] = 0;
  g2[2] = 0;

  g1[3] = K * r *
    (B2 +
    pow(
      a2,
      2) * mb +
    pow(
      c2,
      2) * mb + c2 * mb * r * cos(psi) + (-1) * a2 * mb * r * sin(psi)) *
    (1 / (2 * B2 * Jw + 2 * pow(a2, 2) * Jw * mb + 2 * pow(c2, 2) * Jw * mb + B2 * mb * pow(r, 2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) + 2 * B2 * mw *
    pow(r, 2) + 2 * pow(a2, 2) * mb * mw * pow(r, 2) + 2 * pow(c2, 2) * mb * mw * pow(r, 2) +
    (-1) *
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) *
    pow(
      cos(psi),
      2) + (-1) *
    pow(
      a2,
      2) *
    pow(mb, 2) * pow(r, 2) * pow(sin(psi), 2) + a2 * c2 * pow(mb, 2) * pow(r, 2) * sin(2 * psi)));
  g2[3] = K * r *
    (B2 +
    pow(
      a2,
      2) * mb +
    pow(
      c2,
      2) * mb + c2 * mb * r * cos(psi) + (-1) * a2 * mb * r * sin(psi)) *
    (1 / (2 * B2 * Jw + 2 * pow(a2, 2) * Jw * mb + 2 * pow(c2, 2) * Jw * mb + B2 * mb * pow(r, 2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) + 2 * B2 * mw *
    pow(r, 2) + 2 * pow(a2, 2) * mb * mw * pow(r, 2) + 2 * pow(c2, 2) * mb * mw * pow(r, 2) +
    (-1) *
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) *
    pow(
      cos(psi),
      2) + (-1) *
    pow(
      a2,
      2) *
    pow(mb, 2) * pow(r, 2) * pow(sin(psi), 2) + a2 * c2 * pow(mb, 2) * pow(r, 2) * sin(2 * psi)));

  g1[4] = (-1) * K * L *
    (1 /
    (Jw *
    pow(
      L,
      2) * 1 / r +
    pow(
      L,
      2) * mw * r + 2 *
    (C2 +
    pow(
      a2,
      2) * mb) * r *
    pow(
      cos(psi),
      2) + 2 *
    (A2 + pow(c2, 2) * mb) * r * pow(sin(psi), 2) + 2 * a2 * c2 * mb * r * sin(2 * psi)));
  g2[4] = K * L *
    (1 /
    (Jw *
    pow(
      L,
      2) * 1 / r +
    pow(
      L,
      2) * mw * r + 2 *
    (C2 +
    pow(
      a2,
      2) * mb) * r *
    pow(
      cos(psi),
      2) + 2 *
    (A2 + pow(c2, 2) * mb) * r * pow(sin(psi), 2) + 2 * a2 * c2 * mb * r * sin(2 * psi)));

  g1[5] = 0;
  g2[5] = 0;

  g1[6] = (-2) * K *
    (2 * Jw + mb *
    pow(
      r,
      2) + 2 * mw *
    pow(
      r,
      2) + c2 * mb * r * cos(psi) + (-1) * a2 * mb * r * sin(psi)) *
    (1 /
    (4 * B2 * Jw + 4 * pow(a2, 2) * Jw * mb + 4 * pow(c2, 2) * Jw * mb + 2 * B2 * mb * pow(r, 2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) * pow(mb, 2) * pow(r, 2) + 4 * B2 * mw * pow(r, 2) + 4 * pow(a2, 2) * mb * mw * pow(r, 2) +
    4 *
    pow(
      c2,
      2) * mb * mw *
    pow(
      r,
      2) +
    (pow(
      a2,
      2) + (-1) *
    pow(
      c2,
      2)) *
    pow(mb, 2) * pow(r, 2) * cos(2 * psi) + 2 * a2 * c2 * pow(mb, 2) * pow(r, 2) * sin(2 * psi)));
  g2[6] = (-2) * K *
    (2 * Jw + mb *
    pow(
      r,
      2) + 2 * mw *
    pow(
      r,
      2) + c2 * mb * r * cos(psi) + (-1) * a2 * mb * r * sin(psi)) *
    (1 /
    (4 * B2 * Jw + 4 * pow(a2, 2) * Jw * mb + 4 * pow(c2, 2) * Jw * mb + 2 * B2 * mb * pow(r, 2) +
    pow(
      a2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) +
    pow(
      c2,
      2) *
    pow(
      mb,
      2) *
    pow(
      r,
      2) + 4 * B2 * mw *
    pow(r, 2) + 4 * pow(a2, 2) * mb * mw * pow(r, 2) + 4 * pow(c2, 2) * mb * mw * pow(r, 2) +
    (pow(
      a2,
      2) + (-1) *
    pow(
      c2,
      2)) *
    pow(mb, 2) * pow(r, 2) * cos(2 * psi) + 2 * a2 * c2 * pow(mb, 2) * pow(r, 2) * sin(2 * psi)));


  xDot = f + g * u;
  return xDot;


}

#endif  // SEGWAY_DYNAMICS_HPP_

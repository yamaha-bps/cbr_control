// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef NL_PROBLEM_PENDULUM_HPP_
#define NL_PROBLEM_PENDULUM_HPP_

#include <cbr_math/math.hpp>

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <limits>


template<typename T1, typename T2>
auto pendulum_dynamics(const Eigen::MatrixBase<T1> & x, const Eigen::MatrixBase<T2> & u)
{
  // Define nx, nu
  constexpr auto nx = T1::RowsAtCompileTime;
  constexpr auto nu = T2::RowsAtCompileTime;

  // Initialize Parameters
  constexpr double l = 1.;
  constexpr double m = 1.;
  constexpr double b = 0.2;
  constexpr double gr = 9.81;

  using T = typename decltype(x * u.transpose())::EvalReturnType::Scalar;

  // xDot = f(x) + g(x)*u
  Eigen::Matrix<typename T1::Scalar, nx, 1> f;
  Eigen::Matrix<typename T1::Scalar, nx, nu> g;

  // Define States
  const auto & theta = x[0];
  const auto & thetaDot = x[1];

  // Dynamics
  f[0] = thetaDot;
  f[1] = -(gr / l) * sin(theta) - b / (m * l * l) * thetaDot;
  g[0] = 0.;
  g[1] = 1 / (m * l * l);

  return (f + g * u).eval();
}


struct NlOcpPendulum
{
  constexpr static std::size_t nx = 2;
  constexpr static std::size_t nu = 1;
  constexpr static bool xl_always_feasible = false;

  using state_t = Eigen::Matrix<double, nx, 1>;
  using deriv_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  const Q_t Q = (state_t() << 20.0, 1.0).finished().asDiagonal();
  const Q_t QT = (state_t() << 20.0, 1.0).finished().asDiagonal();
  const R_t R = (R_t() << 0.1).finished();

  template<typename T1, typename T2>
  auto get_f(const Eigen::MatrixBase<T1> & x, const Eigen::MatrixBase<T2> & u) const
  {
    static_assert(
      T1::RowsAtCompileTime == nx &&
      T1::ColsAtCompileTime == 1,
      "First argument must be an nx*1 Eigen Matrix");

    static_assert(
      T2::RowsAtCompileTime == nu &&
      T2::ColsAtCompileTime == 1,
      "Second argument must be an nu*1 Eigen Matrix");

    return pendulum_dynamics<T1, T2>(x, u);
  }

  void get_state_lb(double, Eigen::Ref<state_t> state_lb) const
  {
    const double kInfinity = std::numeric_limits<double>::infinity();
    state_lb <<
      -kInfinity,
      -cbr::deg2rad(200.0);
  }

  void get_state_ub(double, Eigen::Ref<state_t> state_ub) const
  {
    const double kInfinity = std::numeric_limits<double>::infinity();
    state_ub <<
      kInfinity,
      cbr::deg2rad(200.0);
  }

  void get_input_lb(double, Eigen::Ref<input_t> input_lb) const
  {
    input_lb << -0.1;
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub << 0.1;
  }

  const Q_t & get_Q(double) const {return Q;}
  const R_t & get_R(double) const {return R;}
  const Q_t & get_QT() const {return QT;}
};

#endif  // NL_PROBLEM2_HPP_

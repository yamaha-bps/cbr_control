// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef NL_PROBLEM_SEGWAY_HPP_
#define NL_PROBLEM_SEGWAY_HPP_

#include <Eigen/Dense>

#include "segway_dynamics.hpp"


struct SegwayProblem
{
  constexpr static std::size_t nx = 7;
  constexpr static std::size_t nu = 2;
  constexpr static bool xl_always_feasible = false;

  using state_t = Eigen::Matrix<double, nx, 1>;
  using deriv_t = state_t;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // constants x y psi vel r theta thetaDot
  const Q_t Q = (state_t() << 10., 0.001, 0.0, 1., 0.001, 1.0, 0.1).finished().asDiagonal();
  const R_t R = Eigen::Vector2d(1, 0.1).asDiagonal();
  const Q_t QT = Q;

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

    return segway_dynamics<T1, T2>(x, u);
  }

  void get_state_lb(double, Eigen::Ref<state_t> state_lb) const
  {
    state_lb <<
      -100.,
      -100.,
      -6.2832,
      -10.,
      -31.416,
      -6.2832,
      -31.416;
  }

  void get_state_ub(double, Eigen::Ref<state_t> state_ub) const
  {
    state_ub <<
      100.,
      100.,
      6.2832,
      10.,
      31.416,
      6.2832,
      31.416;
  }

  void get_input_lb(double, Eigen::Ref<input_t> input_lb) const
  {
    input_lb << -15, -15;
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub << 15, 15;
  }

  const Q_t & get_Q(double) const {return Q;}
  const R_t & get_R(double) const {return R;}
  const Q_t & get_QT() const {return QT;}

};

#endif  // NL_PROBLEM_SEGWAY_HPP_

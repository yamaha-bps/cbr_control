// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CLTV_PROBLEM_HPP_
#define CLTV_PROBLEM_HPP_

#include <Eigen/Dense>
#include <cstdint>

struct CltvOcpDi
{
  // Must be defined
  constexpr static double T = 1.;
  constexpr static std::size_t nx = 2;
  constexpr static std::size_t nu = 1;

  // Custom types
  using state_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // Constants
  const A_t A = (A_t() << 0., 1., 0., 0.).finished();
  const B_t B = (B_t() << 0., 1.).finished();
  const state_t E = (state_t() << 0., 0.).finished();
  const Q_t Q = (Q_t() << 1000., 0., 0., 10.).finished();
  const Q_t QT = (Q_t() << 100., 0., 0., 100.).finished();
  const R_t R = (R_t() << 0.0001).finished();


  void get_x0(Eigen::Ref<state_t> x0) const
  {
    x0 << 1., 0.;
  }

  void get_T(double & t) const
  {
    t = T;
  }

  void get_state_lb(double, Eigen::Ref<state_t> state_lb) const
  {
    state_lb <<
      -1000.,
      -1.;
  }

  void get_state_ub(double, Eigen::Ref<state_t> state_ub) const
  {
    state_ub <<
      1000.,
      1.;
  }

  void get_input_lb(double, Eigen::Ref<input_t> input_lb) const
  {
    input_lb << -1.;
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub << 1.;
  }

  const A_t & get_A(double) const
  {
    return A;
  }

  const B_t & get_B(double) const
  {
    return B;
  }

  const state_t & get_E(double) const
  {
    return E;
  }

  const Q_t & get_Q(double) const
  {
    return Q;
  }

  const R_t & get_R(double) const
  {
    return R;
  }

  const Q_t & get_QT() const
  {
    return QT;
  }

  state_t get_xldot(double) const {return state_t::Zero();}
};

#endif  // CLTV_PROBLEM_HPP_

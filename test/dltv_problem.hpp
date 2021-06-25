// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef DLTV_PROBLEM_HPP_
#define DLTV_PROBLEM_HPP_

#include <Eigen/Dense>
#include <cstdint>

template<std::size_t _nPts = 100>
struct DltvOcpDi
{
  static_assert(_nPts > 1, "Number of trajectory points must be > 1.");

  // Must be defined
  constexpr static std::size_t nPts = _nPts;
  constexpr static std::size_t nx = 2;
  constexpr static std::size_t nu = 1;

  // Time step
  constexpr static double dt = 0.01;

  // Custom types
  using state_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // Constants
  const A_t A = (A_t() << 1., 0.01, 0., 1.).finished();
  const B_t B = (B_t() << 5.0e-5, 0.01).finished();
  const state_t E = (state_t() << 0., 0.).finished();
  const Q_t Q = (Q_t() << 10., 0., 0., 0.1 ).finished();
  const R_t R = (R_t() << 0.01).finished();
  const Q_t QT = (Q_t() << 1., 0., 0., 1.).finished();

  void get_x0(Eigen::Ref<state_t> x0) const
  {
    x0 << 1., 0.;
  }

  void get_T(std::size_t k, double & t) const
  {
    t = static_cast<double>(k) * dt;
  }

  void get_state_lb(std::size_t, Eigen::Ref<state_t> state_lb) const
  {
    state_lb <<
      -1000.,
      -0.5;
  }

  void get_state_ub(std::size_t, Eigen::Ref<state_t> state_ub) const
  {
    state_ub <<
      1000.,
      0.5;
  }

  void get_input_lb(std::size_t, Eigen::Ref<input_t> input_lb) const
  {
    input_lb << -1;
  }

  void get_input_ub(std::size_t, Eigen::Ref<input_t> input_ub) const
  {
    input_ub << 1;
  }

  const A_t & get_A(std::size_t) const
  {
    return A;
  }

  const B_t & get_B(std::size_t) const
  {
    return B;
  }

  const state_t & get_E(std::size_t) const
  {
    return E;
  }

  const Q_t & get_Q(std::size_t) const
  {
    return Q;
  }

  const R_t & get_R(std::size_t) const
  {
    return R;
  }

  const Q_t & get_QT() const
  {
    return QT;
  }

  state_t get_xldot(double) const {return state_t::Zero();}
};

#endif  // DLTV_PROBLEM_HPP_

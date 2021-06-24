// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef NL_PROBLEM_SE2_HPP_
#define NL_PROBLEM_SE2_HPP_

#include <Eigen/Dense>
#include <cbr_math/lie/Tn.hpp>
#include <cbr_math/lie/group_product.hpp>
#include <sophus/se2.hpp>

/**
 * @brief Defines an optimal control problem on (X, V) \in SO(3) \timex R3 with three inputs
 *   d^r X_t = V
 *   d^r V_t = u
 */
struct SE2Problem
{
  using state_t = cbr::lie::GroupProduct<double, 0, Sophus::SE2, cbr::lie::T3>;
  using deriv_t = typename state_t::Tangent;
  using input_t = Eigen::Vector3d;

  static constexpr std::size_t nx = state_t::DoF;
  static constexpr std::size_t nu = input_t::SizeAtCompileTime;

  // Dynamics (must be differentiable so we make a generic template)
  template<typename T, typename Derived>
  auto get_f(const T & x, const Eigen::MatrixBase<Derived> & u) const
  {
    using Scalar = typename decltype(x.log() * u.transpose())::EvalReturnType::Scalar;

    Eigen::Matrix<Scalar, 6, 1> ret;
    ret.template segment<3>(0) = std::get<1>(x).translation();
    ret.template segment<3>(3) = u.eval();
    return ret;
  }

  void get_input_lb(double, Eigen::Ref<input_t> input_lb) const
  {
    input_lb.setConstant(-0.5);
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub.setConstant(0.5);
  }

  Eigen::Matrix<double, nx, nx> get_Q(double) const
  {
    return 0.1 * Eigen::Matrix<double, nx, nx>::Identity();
  }
  Eigen::Matrix<double, nx, nx> get_QT() const
  {
    return Eigen::Matrix<double, nx, nx>::Identity();
  }
  Eigen::Matrix<double, nu, nu> get_R(double) const
  {
    return 0.1 * Eigen::Matrix<double, nu, nu>::Identity();
  }
};

#endif  // NL_PROBLEM_SE2_HPP_

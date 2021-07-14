// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__CLTV_OCP_LIE_HPP_
#define CBR_CONTROL__MPC__CLTV_OCP_LIE_HPP_

#include <Eigen/Dense>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <cbr_math/lie/common.hpp>
#include <cbr_utils/utils.hpp>

#include <limits>
#include <utility>

#include "ocp_common.hpp"

namespace cbr
{

/* ---------------------------------------------------------------------------------------------- */
/*                          Lie group Optimal Control Problem Linearizer                          */
/* ---------------------------------------------------------------------------------------------- */

/**
 * @brief Linearize a nonlinear control problem defined on a Lie group to a continuous linear time-varying problem
 * @tparam lie_pb_t nonlinear problem on Lie group satisfying interface conditions
 * The difference from linearized trajectory is the Rn state
 *  a(t) = log( Xl(t)^{-1} * X(t) )
 *             <=>
 *  X(t) = Xl(t) * exp (a(t))
 *
 * We have that
 *  \dot a = [d^r exp_a]^{-1} * f(Xl * exp(a), ul + ue) - [d^l exp_a]^{-1} d^r Xl_t
 * which is a nonlinear system on R^{DoF}. Thus the linearized system is
 *  \dot a = A(t) a + B(t) u + E(t) for
 *
 *  A(t) = (d/da) [d^r exp_a]^{-1} * f(Xl * exp(a), ul + ue)   at a = 0, ue = 0
 *  B(t) = (d/du) [d^r exp_a]^{-1} * f(Xl * exp(a), ul + ue)   at a = 0, ue = 0
 *  E(t) = [d^r exp_a]^{-1} f(Xl * exp(a), ul + ue) - [d^l exp_a]^{-1} d^r Xl_t  at a = 0, ue = 0
 *
 * Setting to zero:
 *
 *  A(t) = (d/da) [d^r exp_a]^{-1} * f(Xl * exp(a), ul)  at a = 0
 *  B(t) = (d/u) f(Xl, ul)                               at u = ul
 *  E(t) = f(Xl, ul) - d^r Xl_t
 */
template<typename lie_pb_t>
class CltvOcpLie
{
public:
  using lie_t = typename lie_pb_t::state_t;
  using input_t = typename lie_pb_t::input_t;
  using state_t = typename lie_pb_t::deriv_t;     // linearized problem is defined in tangent space

  static constexpr std::size_t nx = state_t::SizeAtCompileTime;
  static constexpr std::size_t nu = input_t::SizeAtCompileTime;

  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using E_t = Eigen::Matrix<double, nx, 1>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // Get return type of problem functions
  using T_t = std::result_of_t<decltype(&lie_pb_t::get_T)(lie_pb_t)>;
  using x0_t = std::result_of_t<decltype(&lie_pb_t::get_x0)(lie_pb_t)>;
  using xlr_t = std::result_of_t<decltype(&lie_pb_t::get_xl)(lie_pb_t, double)>;
  using ulr_t = std::result_of_t<decltype(&lie_pb_t::get_ul)(lie_pb_t, double)>;
  using xdr_t = std::result_of_t<decltype(&lie_pb_t::get_xd)(lie_pb_t, double)>;
  using udr_t = std::result_of_t<decltype(&lie_pb_t::get_ud)(lie_pb_t, double)>;
  using Qr_t = std::result_of_t<decltype(&lie_pb_t::get_Q)(lie_pb_t, double)>;
  using QTr_t = std::result_of_t<decltype(&lie_pb_t::get_QT)(lie_pb_t)>;
  using Rr_t = std::result_of_t<decltype(&lie_pb_t::get_R)(lie_pb_t, double)>;

  // Check return type of problem functions
  static_assert(
    std::is_same_v<std::decay_t<T_t>, double>,
    "The get_xl method of the problem must return a double.");
  static_assert(
    std::is_same_v<std::decay_t<x0_t>, lie_t>,
    "The get_x0 method of the problem must return the group type (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<xlr_t>, lie_t>,
    "The get_x0 method of the problem must return the group type (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<ulr_t>, input_t>,
    "The get_ul method of the problem must return an nu*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<xdr_t>, lie_t>,
    "The get_x0 method of the problem must return the group type (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<udr_t>, input_t>,
    "The get_ud method of the problem must return an nu*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<Qr_t>, Q_t>,
    "The get_Q method of the problem must return an nx*nx Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<Rr_t>, R_t>,
    "The get_R method of the problem must return an nu*nu Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<QTr_t>, Q_t>,
    "The get_QT method of the problem must return an nx*nx Eigen::Matrix (or a reference to one).");


  // Check problem dimensions
  static_assert(nx > 0, "Number of states must be > 0.");
  static_assert(nu > 0, "Number of inputs must be > 0.");

public:
  CltvOcpLie() = delete;
  CltvOcpLie(const CltvOcpLie &) = default;
  CltvOcpLie(CltvOcpLie &&) noexcept = default;
  CltvOcpLie & operator=(const CltvOcpLie &) = default;
  CltvOcpLie & operator=(CltvOcpLie &&) noexcept = default;
  ~CltvOcpLie() = default;

  explicit CltvOcpLie(const lie_pb_t & pb)
  : nl_pb_(pb)
  {}

  explicit CltvOcpLie(lie_pb_t && pb)
  : nl_pb_(std::move(pb))
  {}

  void get_x0(Eigen::Ref<state_t> x0) const
  {
    x0 = (nl_pb_.get_xl(0.).inverse() * nl_pb_.get_x0()).log();
  }

  void get_T(double & T) const
  {
    T = nl_pb_.get_T();
  }

  void get_state_lb(double, Eigen::Ref<state_t> state_lb) const
  {
    // state bounds not supported for now
    state_lb.setConstant(-std::numeric_limits<double>::infinity());
  }

  void get_state_ub(double, Eigen::Ref<state_t> state_ub) const
  {
    // state bounds not supported for now
    state_ub.setConstant(std::numeric_limits<double>::infinity());
  }

  void get_input_lb(double t, Eigen::Ref<input_t> input_lb) const
  {
    nl_pb_.get_input_lb(t, input_lb);
    input_lb -= nl_pb_.get_ul(t);
  }

  void get_input_ub(double t, Eigen::Ref<input_t> input_ub) const
  {
    nl_pb_.get_input_ub(t, input_ub);
    input_ub -= nl_pb_.get_ul(t);
  }

  A_t get_A(double t) const
  {
    using lie_ad_t = lie::detail::change_scalar_t<lie_t, autodiff::dual>;
    using tangent_ad_t = Eigen::Matrix<autodiff::dual, nx, 1>;

    const lie_ad_t xlin = nl_pb_.get_xl(t).template cast<autodiff::dual>();
    const input_t ulin = nl_pb_.get_ul(t);

    auto fx = [&](const tangent_ad_t & a) -> tangent_ad_t {
        return lie::dr_expinv<lie_ad_t>(a) * nl_pb_.get_f(xlin * lie_ad_t::exp(a), ulin);
      };

    tangent_ad_t a = tangent_ad_t::Zero();
    return autodiff::forward::jacobian(fx, autodiff::wrt(a), autodiff::forward::at(a));
  }

  B_t get_B(double t) const
  {
    using input_ad_t = Eigen::Matrix<autodiff::dual, nu, 1>;
    using tangent_ad_t = Eigen::Matrix<autodiff::dual, nx, 1>;

    const lie_t xlin = nl_pb_.get_xl(t);

    auto fu = [&](const input_ad_t & u) -> tangent_ad_t {
        return nl_pb_.get_f(xlin, u);
      };

    input_ad_t ulin = nl_pb_.get_ul(t);
    return autodiff::forward::jacobian(fu, autodiff::wrt(ulin), autodiff::forward::at(ulin));
  }

  E_t get_E(double t) const
  {
    const lie_t xlin = nl_pb_.get_xl(t);
    const input_t ulin = nl_pb_.get_ul(t);
    const typename lie_t::Tangent xlDot = nl_pb_.get_xldot(t);

    return nl_pb_.get_f(xlin, ulin) - xlDot;
  }

  Qr_t get_Q(double t) const
  {
    return nl_pb_.get_Q(t);
  }

  QTr_t get_QT() const
  {
    return nl_pb_.get_QT();
  }

  Rr_t get_R(double t) const
  {
    return nl_pb_.get_R(t);
  }

  state_t get_q(double t) const
  {
    const xlr_t xl = nl_pb_.get_xl(t);
    const xdr_t xd = nl_pb_.get_xd(t);
    const Qr_t Q = nl_pb_.get_Q(t);
    return (xd.inverse() * xl).log().transpose() * Q;
  }

  state_t get_qT() const
  {
    double T = nl_pb_.get_T();
    const xlr_t xl = nl_pb_.get_xl(T);
    const xdr_t xd = nl_pb_.get_xd(T);
    const QTr_t QT = nl_pb_.get_QT();
    return (xd.inverse() * xl).log().transpose() * QT;
  }

  input_t get_r(double t) const
  {
    const ulr_t ul = nl_pb_.get_ul(t);
    const udr_t ud = nl_pb_.get_ud(t);
    const Rr_t R = nl_pb_.get_R(t);
    return (ul - ud).transpose() * R;
  }

  lie_pb_t & problem()
  {
    return nl_pb_;
  }

protected:
  lie_pb_t nl_pb_{};
};

// Class template argument deduction guides
template<typename T>
CltvOcpLie(T)->CltvOcpLie<T>;

template<typename T1, typename T2>
CltvOcpLie(T1, T2)->CltvOcpLie<T1>;

}  // namespace cbr


#endif  // CBR_CONTROL__MPC__CLTV_OCP_LIE_HPP_

// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__CLTV_OCP_HPP_
#define CBR_CONTROL__MPC__CLTV_OCP_HPP_

#include <Eigen/Dense>

#include <cbr_utils/utils.hpp>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <utility>

#include "ocp_common.hpp"

namespace cbr
{

/* ---------------------------------------------------------------------------------------------- */
/*                          Nonlinear Optimal Control Problem Linearizer                          */
/* ---------------------------------------------------------------------------------------------- */

struct CltvOcpParams
{
};

/**
 * @brief Linearize a nonlinear control problem to a continuous linear time-varying problem
 * @tparam nl_pb_t nonlinear problem satisfying interface conditions
 */
template<typename nl_pb_t>
class CltvOcp
{
public:
  // Must be defined in nl_pb_t problem
  constexpr static std::size_t nx = nl_pb_t::nx;
  constexpr static std::size_t nu = nl_pb_t::nu;

  // Create some useful aliases
  using state_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // Get return type of problem functions
  using Tr_t = std::result_of_t<decltype(&nl_pb_t::get_T)(nl_pb_t)>;
  using x0r_t = std::result_of_t<decltype(&nl_pb_t::get_x0)(nl_pb_t)>;
  using xlr_t = std::result_of_t<decltype(&nl_pb_t::get_xl)(nl_pb_t, double)>;
  using ulr_t = std::result_of_t<decltype(&nl_pb_t::get_ul)(nl_pb_t, double)>;
  using xdr_t = std::result_of_t<decltype(&nl_pb_t::get_xd)(nl_pb_t, double)>;
  using udr_t = std::result_of_t<decltype(&nl_pb_t::get_ud)(nl_pb_t, double)>;
  using Qr_t = std::result_of_t<decltype(&nl_pb_t::get_Q)(nl_pb_t, double)>;
  using QTr_t = std::result_of_t<decltype(&nl_pb_t::get_QT)(nl_pb_t)>;
  using Rr_t = std::result_of_t<decltype(&nl_pb_t::get_R)(nl_pb_t, double)>;

  // Check return type of problem functions
  static_assert(
    std::is_same_v<std::decay_t<Tr_t>, double>,
    "The get_T method of the problem must return a double (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<x0r_t>, state_t>,
    "The get_x0 method of the problem must return an nx*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<xlr_t>, state_t>,
    "The get_xl method of the problem must return an nx*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<ulr_t>, input_t>,
    "The get_ul method of the problem must return an nu*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<xdr_t>, state_t>,
    "The get_xd method of the problem must return an nx*1 Eigen::Matrix (or a reference to one).");
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
  CltvOcp() = delete;
  CltvOcp(const CltvOcp &) = default;
  CltvOcp(CltvOcp &&) = default;
  CltvOcp & operator=(const CltvOcp &) = default;
  CltvOcp & operator=(CltvOcp &&) = default;

  explicit CltvOcp(const nl_pb_t & pb)
  : nl_pb_(pb)
  {}

  explicit CltvOcp(nl_pb_t && pb)
  : nl_pb_(std::move(pb))
  {}

  template<typename T1, typename T2>
  CltvOcp(T1 && pb, T2 && prm)
  : nl_pb_(std::forward<T1>(pb)),
    prm_(std::forward<T2>(prm))
  {}

  void get_x0(Eigen::Ref<state_t> x0) const
  {
    x0 = nl_pb_.get_x0();
    x0 -= nl_pb_.get_xl(0.);
  }

  void get_T(double & t) const
  {
    t = nl_pb_.get_T();
  }

  void get_state_lb(double t, Eigen::Ref<state_t> state_lb) const
  {
    nl_pb_.get_state_lb(t, state_lb);
    state_lb -= nl_pb_.get_xl(t);
  }

  void get_state_ub(double t, Eigen::Ref<state_t> state_ub) const
  {
    nl_pb_.get_state_ub(t, state_ub);
    state_ub -= nl_pb_.get_xl(t);
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
    using X_t = Eigen::Matrix<autodiff::dual, nx, 1>;
    X_t xlin = nl_pb_.get_xl(t);
    const ulr_t u = nl_pb_.get_ul(t);

    auto fx = [&](const X_t & x) -> X_t {
        return nl_pb_.get_f(x, u);
      };

    return autodiff::forward::jacobian(fx, autodiff::wrt(xlin), autodiff::forward::at(xlin));
  }

  B_t get_B(double t) const
  {
    using X_t = Eigen::Matrix<autodiff::dual, nx, 1>;
    using U_t = Eigen::Matrix<autodiff::dual, nu, 1>;
    U_t ulin = nl_pb_.get_ul(t);
    const xlr_t x = nl_pb_.get_xl(t);

    auto fu = [&](const U_t & u) -> X_t {
        return nl_pb_.get_f(x, u);
      };

    return autodiff::forward::jacobian(fu, autodiff::wrt(ulin), autodiff::forward::at(ulin));
  }

  state_t get_E(double t) const
  {
    using xldotr_t = std::result_of_t<decltype(&nl_pb_t::get_xldot)(nl_pb_t, double)>;

    const xlr_t xl = nl_pb_.get_xl(t);
    const xldotr_t xlDot = nl_pb_.get_xldot(t);
    const ulr_t ul = nl_pb_.get_ul(t);
    return nl_pb_.get_f(xl, ul) - xlDot;
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
    return (xl - xd).transpose() * Q;
  }

  state_t get_qT() const
  {
    double T = nl_pb_.get_T();
    const xlr_t xl = nl_pb_.get_xl(T);
    const xdr_t xd = nl_pb_.get_xd(T);
    const QTr_t QT = nl_pb_.get_QT();
    return (xl - xd).transpose() * QT;
  }

  input_t get_r(double t) const
  {
    const ulr_t ul = nl_pb_.get_ul(t);
    const udr_t ud = nl_pb_.get_ud(t);
    const Rr_t R = nl_pb_.get_R(t);
    return (ul - ud).transpose() * R;
  }

  template<typename T>
  void set_params(T && p)
  {
    prm_ = std::forward<T>(p);
  }

  nl_pb_t & problem()
  {
    return nl_pb_;
  }

protected:
  nl_pb_t nl_pb_{};
  CltvOcpParams prm_{};
};

// Class template argument deduction guides
template<typename T>
CltvOcp(T)->CltvOcp<T>;

template<typename T1, typename T2>
CltvOcp(T1, T2)->CltvOcp<T1>;

}  // namespace cbr

#endif  // CBR_CONTROL__MPC__CLTV_OCP_HPP_

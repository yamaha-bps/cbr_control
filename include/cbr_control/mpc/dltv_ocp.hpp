// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__DLTV_OCP_HPP_
#define CBR_CONTROL__MPC__DLTV_OCP_HPP_

#include <Eigen/Dense>

#include <cbr_utils/utils.hpp>

#include <utility>

#include "ocp_common.hpp"

namespace cbr
{

/* ---------------------------------------------------------------------------------------------- */
/*               Continuous Time Varying Linear Optimal Control Problem Discretizer               */
/* ---------------------------------------------------------------------------------------------- */

template<typename cltv_pb_t, std::size_t _nPts = 100, std::size_t exp_order = 4>
class DltvOcp
{
public:
  // Must be defined in ctlv_pb problem
  constexpr static std::size_t nx = cltv_pb_t::nx;
  constexpr static std::size_t nu = cltv_pb_t::nu;
  constexpr static std::size_t nPts = _nPts;

  using problem_t = cltv_pb_t;

  // Create some useful aliases
  using state_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using time_t = Eigen::Matrix<double, nPts, 1>;
  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  // Get return type of problem functions
  using Ar_t = std::result_of_t<decltype(&cltv_pb_t::get_A)(cltv_pb_t, double)>;
  using Br_t = std::result_of_t<decltype(&cltv_pb_t::get_B)(cltv_pb_t, double)>;
  using Qr_t = std::result_of_t<decltype(&cltv_pb_t::get_Q)(cltv_pb_t, double)>;
  using QTr_t = std::result_of_t<decltype(&cltv_pb_t::get_QT)(cltv_pb_t)>;
  using Rr_t = std::result_of_t<decltype(&cltv_pb_t::get_R)(cltv_pb_t, double)>;

  /* -------------------------------------------------------------------------- */
  /*                                  Optionals                                 */
  /* -------------------------------------------------------------------------- */

  // Check existance of get_E function
  constexpr static bool has_E_approx = std::experimental::is_detected_v<
    ocp_detail::has_E_continuous, cltv_pb_t>;
  constexpr static bool has_E = std::experimental::is_detected_exact_v<
    state_t, ocp_detail::has_E_continuous, cltv_pb_t>||
    std::experimental::is_detected_exact_v<
    const state_t &, ocp_detail::has_E_continuous, cltv_pb_t>;
  using Er_t = std::experimental::detected_or_t<
    state_t, ocp_detail::has_E_continuous, cltv_pb_t>;
  static_assert(
    !(has_E_approx && !has_E),
    "Detected get_E function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_q function
  constexpr static bool has_q_approx = std::experimental::is_detected_v<
    ocp_detail::has_q_continuous, cltv_pb_t>;
  constexpr static bool has_q = std::experimental::is_detected_exact_v<
    state_t, ocp_detail::has_q_continuous, cltv_pb_t>||
    std::experimental::is_detected_exact_v<
    const state_t &, ocp_detail::has_q_continuous, cltv_pb_t>;
  using qr_t = std::experimental::detected_or_t<
    state_t, ocp_detail::has_q_continuous, cltv_pb_t>;
  static_assert(
    !(has_q_approx && !has_q),
    "Detected get_q function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_qT function
  constexpr static bool has_qT_approx = std::experimental::is_detected_v<
    ocp_detail::has_qT_continuous, cltv_pb_t>;
  constexpr static bool has_qT = std::experimental::is_detected_exact_v<
    state_t, ocp_detail::has_qT_continuous, cltv_pb_t>||
    std::experimental::is_detected_exact_v<
    const state_t &, ocp_detail::has_qT_continuous, cltv_pb_t>;

  using qTr_t = std::experimental::detected_or_t<
    state_t, ocp_detail::has_qT_continuous, cltv_pb_t>; \
  static_assert(
    !(has_qT_approx && !has_qT),
    "Detected get_qT function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_r function
  constexpr static bool has_r_approx = std::experimental::is_detected_v<
    ocp_detail::has_r_continuous, cltv_pb_t>;
  constexpr static bool has_r = std::experimental::is_detected_exact_v<
    input_t, ocp_detail::has_r_continuous, cltv_pb_t>||
    std::experimental::is_detected_exact_v<
    const input_t &, ocp_detail::has_r_continuous, cltv_pb_t>;
  using rr_t = std::experimental::detected_or_t<
    input_t, ocp_detail::has_r_continuous, cltv_pb_t>;
  static_assert(
    !(has_r_approx && !has_r),
    "Detected get_r function doesn't have a correct return type. "
    "It must be an nu*1 Eigen::Matrix (or a const reference to one)");

  // Check problem dimensions
  static_assert(nx > 0, "Number of states must be > 0.");
  static_assert(nu > 0, "Number of inputs must be > 0.");
  static_assert(nPts > 1, "Number of trajectory points must be > 1.");
  static_assert(exp_order < 20, "Exponential order must be < 20.");

  // Check return type of problem functions
  static_assert(
    std::is_same_v<std::decay_t<Ar_t>, A_t>,
    "The get_A method of the problem must return an nx*nx Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<Br_t>, B_t>,
    "The get_B method of the problem must return an nx*nu Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<Qr_t>, Q_t>,
    "The get_Q method of the problem must return an nx*nx Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<qr_t>, state_t>,
    "The get_q method of the problem must return an nx*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<QTr_t>, Q_t>,
    "The get_QT method of the problem must return an nx*nx Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<qTr_t>, state_t>,
    "The get_qT method of the problem must return an nx*1 Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<Rr_t>, R_t>,
    "The get_R method of the problem must return an nu*nu Eigen::Matrix (or a reference to one).");
  static_assert(
    std::is_same_v<std::decay_t<rr_t>, input_t>,
    "The get_r method of the problem must return an nu*1 Eigen::Matrix (or a reference to one).");

public:
  DltvOcp() = delete;
  DltvOcp(const DltvOcp &) = default;
  DltvOcp(DltvOcp &&) = default;
  DltvOcp & operator=(const DltvOcp &) = default;
  DltvOcp & operator=(DltvOcp &&) = default;

  explicit DltvOcp(const cltv_pb_t & pb)
  : cltv_pb_(pb),
    dt_{compute_dt()}
  {}

  explicit DltvOcp(cltv_pb_t && pb)
  : cltv_pb_(std::move(pb)),
    dt_{compute_dt()}
  {}

  template<typename T1, typename T2>
  DltvOcp(T1 && pb, T2 && prm)
  : cltv_pb_(std::forward<T1>(pb)),
    dt_{compute_dt()}
  {}

  void get_x0(Eigen::Ref<state_t> x0) const
  {
    cltv_pb_.get_x0(x0);
  }

  void get_state_lb(std::size_t k, Eigen::Ref<state_t> state_lb) const
  {
    cltv_pb_.get_state_lb(indexToTime(k), state_lb);
  }

  void get_state_ub(std::size_t k, Eigen::Ref<state_t> state_ub) const
  {
    cltv_pb_.get_state_ub(indexToTime(k), state_ub);
  }

  void get_input_lb(std::size_t k, Eigen::Ref<input_t> input_lb) const
  {
    cltv_pb_.get_input_lb(indexToTime(k), input_lb);
  }

  void get_input_ub(std::size_t k, Eigen::Ref<input_t> input_ub) const
  {
    cltv_pb_.get_input_ub(indexToTime(k), input_ub);
  }

  A_t get_A(std::size_t k) const
  {
    // define identity for order = 0
    A_t expA = A_t::Identity();

    if constexpr (exp_order > 0) {
      const A_t Adt = dt_ * cltv_pb_.get_A(indexToTime(k));
      expA += Adt;
      if constexpr (exp_order > 1) {
        double c = 1.;
        A_t Adtp = Adt;
        for (std::size_t i = 2; i <= exp_order; i++) {
          Adtp *= Adt;
          c /= static_cast<double>(i);
          expA += c * Adtp;
        }
      }
    }

    return expA;
  }

  B_t get_B(std::size_t k) const
  {
    // define identity for order = 0
    A_t expA = A_t::Identity();

    if constexpr (exp_order > 0) {
      const A_t Adt = dt_ * cltv_pb_.get_A(indexToTime(k));
      expA += Adt / 2.;
      if constexpr (exp_order > 1) {
        double c = 0.5;
        A_t Adtp = Adt / 2.;
        for (std::size_t i = 2; i <= exp_order; i++) {
          Adtp *= Adt;
          c /= static_cast<double>(i + 1);
          expA += c * Adtp;
        }
      }
    }

    B_t expB = expA * cltv_pb_.get_B(indexToTime(k)) * dt_;

    return expB;
  }

  template<typename T = std::size_t>
  state_t get_E(std::enable_if_t<has_E, T> k)
  {
    // define identity for order = 0
    A_t expA = A_t::Identity();

    if constexpr (exp_order > 0) {
      const A_t Adt = dt_ * cltv_pb_.get_A(indexToTime(k));
      expA += Adt / 2.;
      if constexpr (exp_order > 1) {
        double c = 0.5;
        A_t Adtp = Adt / 2.;
        for (std::size_t i = 2; i <= exp_order; i++) {
          Adtp *= Adt;
          c /= static_cast<double>(i + 1);
          expA += c * Adtp;
        }
      }
    }

    state_t expE = expA * cltv_pb_.get_E(indexToTime(k)) * dt_;
    return expE;
  }

  Q_t get_Q(std::size_t k) const
  {
    return dt_ * cltv_pb_.get_Q(indexToTime(k));
  }

  template<typename T = std::size_t>
  state_t get_q(std::enable_if_t<has_q, T> k) const
  {
    return dt_ * cltv_pb_.get_q(indexToTime(k));
  }

  R_t get_R(std::size_t k) const
  {
    return dt_ * cltv_pb_.get_R(indexToTime(k));
  }

  template<typename T = std::size_t>
  input_t get_r(std::enable_if_t<has_r, T> k) const
  {
    return dt_ * cltv_pb_.get_r(indexToTime(k));
  }

  Q_t get_QT() const
  {
    double T;
    cltv_pb_.get_T(T);
    return cltv_pb_.get_QT();
  }

  template<typename T = void *>
  state_t get_qT([[maybe_unused]] std::enable_if_t<has_qT, T> k = nullptr) const
  {
    double TT;
    cltv_pb_.get_T(TT);
    return cltv_pb_.get_qT();
  }


  cltv_pb_t & problem()
  {
    return cltv_pb_;
  }

  double indexToTime(std::size_t k) const
  {
    return static_cast<double>(k) * dt_;
  }

protected:
  double compute_dt()
  {
    double T;
    cltv_pb_.get_T(T);
    return T / static_cast<double>(nPts - 1);
  }

protected:
  cltv_pb_t cltv_pb_{};
  double dt_{};
};

// Class template argument deduction guides
template<typename T>
DltvOcp(T)->DltvOcp<T>;

template<typename T1, typename T2>
DltvOcp(T1, T2)->DltvOcp<T1, T2::nPts, T2::expOrder>;

}  // namespace cbr


#endif  // CBR_CONTROL__MPC__DLTV_OCP_HPP_

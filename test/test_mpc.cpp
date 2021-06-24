// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#include <gtest/gtest.h>

#include <cbr_math/lie/group_product.hpp>
#include <cbr_math/lie/Tn.hpp>

#include <cbr_control/mpc/dltv_ocp_solver.hpp>
#include <cbr_control/mpc/dltv_ocp.hpp>
#include <cbr_control/mpc/cltv_ocp.hpp>
#include <cbr_control/mpc/cltv_ocp_lie.hpp>
#include <cbr_control/mpc/ocp_common.hpp>

#include <cbr_math/math.hpp>

#include <utility>
#include <limits>

#include "cltv_problem.hpp"
#include "dltv_problem.hpp"


TEST(MPC, solver)
{
  using cbr::DltvOcpSolver;
  using cbr::DltvOcpSolverParams;

  constexpr std::size_t nPts = 3;
  // Instantiate object "pb" (problem) of type TestProblem
  DltvOcpDi<nPts> pb;

  // Instantiate "solver" of type DltvOcpSolver using object "Problem"
  DltvOcpSolverParams prm{};
  prm.osqp_settings.verbose = false;
  DltvOcpSolver solver(std::move(pb), std::move(prm));
  solver.init();

  // Set Solver Parameter
  const auto & sol = solver.solve();
  ASSERT_EQ(sol.rc, cbr::DltvOcpSolverCode::success);
}

TEST(MPC, discretizer)
{
  constexpr double eps = 1e-8;

  using cbr::DltvOcp;
  using cbr::DltvOcpParams;
  using cbr::DltvOcpSolver;
  using cbr::DltvOcpSolverParams;

  constexpr std::size_t nPts = 101;
  DltvOcpDi<nPts> dpb_ref;

  CltvOcpDi cpb;
  DltvOcp<CltvOcpDi, nPts, 4> dpb{std::move(cpb)};

  const auto expA = dpb.get_A(0);
  const auto expB = dpb.get_B(0);

  const auto expA_ref = dpb_ref.get_A(0);
  const auto expB_ref = dpb_ref.get_B(0);

  for (std::size_t i = 0; i < 2 * 2; i++) {
    ASSERT_NEAR(*(expA.data() + i), *(expA_ref.data() + i), eps);
  }

  for (std::size_t i = 0; i < 2 * 1; i++) {
    ASSERT_NEAR(*(expB.data() + i), *(expB_ref.data() + i), eps);
  }

  DltvOcpSolverParams prm{};
  prm.osqp_settings.verbose = false;
  DltvOcpSolver solver{std::move(dpb), std::move(prm)};
  solver.init();

  const auto & sol = solver.solve();
  ASSERT_EQ(sol.rc, cbr::DltvOcpSolverCode::success);
}

struct nl_pb_t
{
  constexpr static std::size_t nx = 1;
  constexpr static std::size_t nu = 1;

  using state_t = Eigen::Matrix<double, nx, 1>;
  using deriv_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;

  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;

  const Q_t Q = (state_t() << 20.0).finished().asDiagonal();
  const Q_t QT = (state_t() << 20.0).finished().asDiagonal();
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

    return (x * x + u).eval();
  }

  void get_state_lb(double, Eigen::Ref<state_t> state_lb) const
  {
    const double kInfinity = std::numeric_limits<double>::infinity();
    state_lb <<
      -kInfinity;
  }

  void get_state_ub(double, Eigen::Ref<state_t> state_ub) const
  {
    const double kInfinity = std::numeric_limits<double>::infinity();
    state_ub <<
      kInfinity;
  }

  void get_input_lb(double, Eigen::Ref<input_t> input_lb) const
  {
    input_lb << -0.1;
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub << 0.1;
  }


  double get_T() const
  {
    return 5;
  }

  state_t get_x0() const
  {
    return state_t{};
  }


  const Q_t & get_Q(double) const {return Q;}
  const R_t & get_R(double) const {return R;}
  const Q_t & get_QT() const {return QT;}

  state_t get_xl(double) const
  {
    return state_t{};
  }
  state_t get_xd(double) const
  {
    return state_t{};
  }
  deriv_t get_xldot(double) const
  {
    return deriv_t::Zero();
  }

  input_t get_ul(double) const
  {
    return input_t::Zero();
  }
  input_t get_ud(double) const
  {
    return input_t::Zero();
  }
};

TEST(MPC, linearize) {
  nl_pb_t ocp{};
  cbr::CltvOcp<nl_pb_t> ocp_lin{ocp};
  auto A = ocp_lin.get_A(1.);
  Eigen::Matrix<double, 1, 1> A_exact = Eigen::Matrix<double, 1, 1>::Zero();
  A_exact[0] = 0.;
  ASSERT_LE((A - A_exact).norm(), 1e-5);
}

struct se2_pb_t
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
    input_lb.setConstant(-1);
  }

  void get_input_ub(double, Eigen::Ref<input_t> input_ub) const
  {
    input_ub.setConstant(1);
  }

  double get_T() const
  {
    return 5;
  }

  state_t get_x0() const
  {
    return state_t{};
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

  state_t get_xl(double) const
  {
    state_t ret{};
    std::get<0>(ret) = Sophus::SE2d::rot(th);
    std::get<1>(ret).translation().x() = vx;
    std::get<1>(ret).translation().y() = vy;
    std::get<1>(ret).translation().z() = wZ;
    return ret;
  }
  state_t get_xd(double) const
  {
    return state_t{};
  }
  deriv_t get_xldot(double) const
  {
    return deriv_t::Zero();
  }

  input_t get_ul(double) const
  {
    return input_t::Zero();
  }
  input_t get_ud(double) const
  {
    return input_t::Zero();
  }

  double th, vx, vy, wZ;
};


TEST(MPC, LieLinearize)
{
  se2_pb_t ocp{};
  ocp.th = 0.0;
  ocp.vx = 1;
  ocp.vy = 0.1;
  ocp.wZ = 0.2;
  cbr::CltvOcpLie<se2_pb_t> ocp_lin{ocp};

  auto A = ocp_lin.get_A(1);

  Eigen::Matrix<double, 6, 6> A_exact;
  A_exact.setZero();

  A_exact(0, 1) = ocp.wZ / 2;
  A_exact(1, 0) = -ocp.wZ / 2;
  A_exact(0, 2) = -ocp.vy / 2;
  A_exact(1, 2) = ocp.vx / 2;

  A_exact.topRightCorner<3, 3>().setIdentity();

  ASSERT_LE((A - A_exact).norm(), 1e-5);
}

template class cbr::DltvOcpSolver<cbr::DltvOcp<cbr::CltvOcpLie<se2_pb_t>>>;
template class cbr::DltvOcpSolver<cbr::DltvOcp<cbr::CltvOcp<nl_pb_t>>>;

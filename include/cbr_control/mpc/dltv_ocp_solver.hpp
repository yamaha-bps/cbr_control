// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__DLTV_OCP_SOLVER_HPP_
#define CBR_CONTROL__MPC__DLTV_OCP_SOLVER_HPP_

#include <Eigen/Dense>

#include <cbr_utils/utils.hpp>

#include <utility>
#include <array>

#include "cbr_control/osqp-cpp.hpp"
#include "ocp_common.hpp"

namespace cbr
{

namespace ocp_detail
{
template<std::size_t nx, std::size_t nu, std::size_t nPts>
constexpr auto DltvOcpSolverCostSparsity()  // objective
{
  constexpr std::size_t nz = (nx + nu) * (nPts - 1);
  std::array<std::size_t, nz> out{};

  std::array<std::size_t, nx> etaX{};
  std::array<std::size_t, nx> etaU{};

  // Generate original vectors
  for (std::size_t i = 0; i < nx; i++) {
    etaX[i] = i + 1;
  }
  for (std::size_t i = 0; i < nu; i++) {
    etaU[i] = i + 1;
  }

  // populate LHS values
  for (std::size_t j = 0; j < nPts - 1; j++) {
    for (std::size_t i = 0; i < nx; i++) {
      out[(j * nx) + i] = etaX[i];
    }
  }

  std::size_t oft = nx * (nPts - 1);
  // populate RHS values
  for (std::size_t j = 0; j < nPts - 1; j++) {
    for (std::size_t i = 0; i < nu; i++) {
      out[oft + (j * nu) + i] = etaU[i];
    }
  }

  return out;
}

template<std::size_t nx, std::size_t nu, std::size_t nPts>
constexpr auto DltvOcpSolverCstrSparsity()  // constraint
{
  constexpr std::size_t nz = (nx + nu) * (nPts - 1);
  std::array<std::size_t, nz> out{};

  std::array<std::size_t, nx> etaX{};
  std::array<std::size_t, nx> etaU{};

  // Generate original vectors
  for (std::size_t i = 0; i < nx; i++) {
    etaX[i] = nx + 2;
  }

  for (std::size_t i = 0; i < nu; i++) {
    etaU[i] = nx + 1;
  }

  // populate LHS values
  for (std::size_t j = 0; j < nPts - 1; j++) {
    for (std::size_t i = 0; i < nx; i++) {
      if (j == nPts - 2) {
        out[(j * nx) + i] = 2;
      } else {
        out[(j * nx) + i] = etaX[i];
      }
    }
  }

  std::size_t oft = nx * (nPts - 1);
  // populate RHS values
  for (std::size_t j = 0; j < nPts - 1; j++) {
    for (std::size_t i = 0; i < nu; i++) {
      out[oft + (j * nu) + i] = etaU[i];
    }
  }

  return out;
}

}  // namespace ocp_detail

/* ---------------------------------------------------------------------------------------------- */
/*                Discrete Time Linear Time Varying Optimal Control Problem Solver                */
/* ---------------------------------------------------------------------------------------------- */

struct DltvOcpSolverParams
{
  osqp::OsqpSettings osqp_settings{};
};

// return code
enum class DltvOcpSolverCode
{
  no_run,
  success,
  failure,
};

/**
 * @brief Solve a linear optimal control problem of type
 *
 *   min_{x, u}   \sum_{k=0}^{K-1} [(1/2) x_k' Q x_k + q' x+k  + (1/2) u_k' R u_k + r' u_k ]  +  (1/2) x_K' QT Q_K
 *
 *    s.t.        x_{k+1} = A_k x_k + B_k u_k + E_k
 *
 * The optimal control problem is defined by implementing get_xxx methods as seen below
 *
 * The variable vector is
 *
 *  [x_1, x_2, \ldots, x_K, u_0, u_1, \ldots, u_{K-1}]
 *
 * @tparam dltv_pb_t
 */
template<typename dltv_pb_t>
class DltvOcpSolver
{
public:
  // Must be defined in dtlv_pb problem
  constexpr static std::size_t nx = dltv_pb_t::nx;
  constexpr static std::size_t nu = dltv_pb_t::nu;
  constexpr static std::size_t nPts = dltv_pb_t::nPts;

  using problem_t = dltv_pb_t;

  // Variable sizes
  constexpr static std::size_t nX = (nPts - 1) * nx;
  constexpr static std::size_t nU = (nPts - 1) * nu;
  constexpr static std::size_t nZ = nX + nU;

  // Constraint matrix sizes
  constexpr static std::size_t nIneqCstr = nZ;
  constexpr static std::size_t nEqCstr = nX;
  constexpr static std::size_t nCstr = nIneqCstr + nEqCstr;

  // QP sparsity
  constexpr static std::array<std::size_t, nZ> costSparsity =
    ocp_detail::DltvOcpSolverCostSparsity<nx, nu, nPts>();
  constexpr static std::array<std::size_t, nZ> cstrSparsity =
    ocp_detail::DltvOcpSolverCstrSparsity<nx, nu, nPts>();

  // Create some useful aliases
  using state_t = Eigen::Matrix<double, nx, 1>;
  using input_t = Eigen::Matrix<double, nu, 1>;
  using A_t = Eigen::Matrix<double, nx, nx>;
  using B_t = Eigen::Matrix<double, nx, nu>;
  using Q_t = Eigen::Matrix<double, nx, nx>;
  using R_t = Eigen::Matrix<double, nu, nu>;
  using state_traj_t = Eigen::Matrix<double, nx, nPts>;
  using input_traj_t = Eigen::Matrix<double, nu, nPts>;
  using z_t = Eigen::Matrix<double, nZ, 1>;

  // Get return type of problem functions
  using Ar_t = std::result_of_t<decltype(&dltv_pb_t::get_A)(dltv_pb_t, std::size_t)>;
  using Br_t = std::result_of_t<decltype(&dltv_pb_t::get_B)(dltv_pb_t, std::size_t)>;
  using Qr_t = std::result_of_t<decltype(&dltv_pb_t::get_Q)(dltv_pb_t, std::size_t)>;
  using QTr_t = std::result_of_t<decltype(&dltv_pb_t::get_QT)(dltv_pb_t)>;
  using Rr_t = std::result_of_t<decltype(&dltv_pb_t::get_R)(dltv_pb_t, std::size_t)>;

  /* -------------------------------------------------------------------------- */
  /*                                  Optionals                                 */
  /* -------------------------------------------------------------------------- */

  // Check existance of get_E function
  constexpr static bool has_E_approx = std::experimental::is_detected_v<ocp_detail::has_E_discrete,
      dltv_pb_t>;
  constexpr static bool has_E = std::experimental::is_detected_exact_v<state_t,
      ocp_detail::has_E_discrete, dltv_pb_t>||
    std::experimental::is_detected_exact_v<const state_t &, ocp_detail::has_E_discrete, dltv_pb_t>;
  using Er_t = std::experimental::detected_or_t<state_t, ocp_detail::has_E_discrete, dltv_pb_t>;
  static_assert(
    !(has_E_approx && !has_E),
    "Detected get_E function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_q function
  constexpr static bool has_q_approx = std::experimental::is_detected_v<ocp_detail::has_q_discrete,
      dltv_pb_t>;
  constexpr static bool has_q = std::experimental::is_detected_exact_v<state_t,
      ocp_detail::has_q_discrete, dltv_pb_t>||
    std::experimental::is_detected_exact_v<const state_t &, ocp_detail::has_q_discrete, dltv_pb_t>;
  using qr_t = std::experimental::detected_or_t<state_t, ocp_detail::has_q_discrete, dltv_pb_t>;
  static_assert(
    !(has_q_approx && !has_q),
    "Detected get_q function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_qT function
  constexpr static bool has_qT_approx = std::experimental::is_detected_v<
    ocp_detail::has_qT_discrete, dltv_pb_t>;
  constexpr static bool has_qT = std::experimental::is_detected_exact_v<state_t,
      ocp_detail::has_qT_discrete, dltv_pb_t>||
    std::experimental::is_detected_exact_v<const state_t &, ocp_detail::has_qT_discrete, dltv_pb_t>;
  using qTr_t = std::experimental::detected_or_t<state_t, ocp_detail::has_qT_discrete, dltv_pb_t>;
  static_assert(
    !(has_qT_approx && !has_qT),
    "Detected get_qT function doesn't have a correct return type. "
    "It must be an nx*1 Eigen::Matrix (or a const reference to one)");

  // Check existance of get_r function
  constexpr static bool has_r_approx = std::experimental::is_detected_v<ocp_detail::has_r_discrete,
      dltv_pb_t>;
  constexpr static bool has_r = std::experimental::is_detected_exact_v<input_t,
      ocp_detail::has_r_discrete, dltv_pb_t>||
    std::experimental::is_detected_exact_v<const input_t &, ocp_detail::has_r_discrete, dltv_pb_t>;
  using rr_t = std::experimental::detected_or_t<input_t, ocp_detail::has_r_discrete, dltv_pb_t>;
  static_assert(
    !(has_r_approx && !has_r),
    "Detected get_r function doesn't have a correct return type. "
    "It must be an nu*1 Eigen::Matrix (or a const reference to one)");

  // Check problem dimensions
  static_assert(nx > 0, "Number of states must be > 0.");
  static_assert(nu > 0, "Number of inputs must be > 0.");
  static_assert(nPts > 1, "Number of trajectory points must be > 1.");

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
  // define structure SOLUTION for the output of the SOLVER
  struct Solution
  {
    DltvOcpSolverCode rc = DltvOcpSolverCode::no_run;
    state_traj_t x = state_traj_t::Zero();
    input_traj_t u = input_traj_t::Zero();
    z_t z = z_t::Zero();
  };

public:
  DltvOcpSolver() = delete;
  DltvOcpSolver(const DltvOcpSolver &) = default;
  DltvOcpSolver(DltvOcpSolver &&) = default;
  DltvOcpSolver & operator=(const DltvOcpSolver &) = default;
  DltvOcpSolver & operator=(DltvOcpSolver &&) = default;

  explicit DltvOcpSolver(const dltv_pb_t & pb)
  : dltv_pb_(pb) {}

  explicit DltvOcpSolver(dltv_pb_t && pb)
  : dltv_pb_(std::move(pb)) {}

  template<typename T1, typename T2>
  DltvOcpSolver(T1 && pb, T2 && prm)
  : dltv_pb_(std::forward<T1>(pb)),
    prm_(std::forward<T2>(prm)) {}

public:
  /**
   * @brief Read problem information and build constraint matrices
   */
  void init()
  {
    osqp::OsqpInstance osqp_instance;

    // Resize osqp_instance matrices
    osqp_instance.objective_matrix.resize(nZ, nZ);
    osqp_instance.objective_vector.resize(nZ);
    osqp_instance.constraint_matrix.resize(nCstr, nZ);
    osqp_instance.lower_bounds.resize(nCstr);
    osqp_instance.upper_bounds.resize(nCstr);

    // Set sparsity structure
    const Eigen::Map<const Eigen::Matrix<std::size_t, nZ, 1>> costSparsityVec(costSparsity.data());
    const Eigen::Map<const Eigen::Matrix<std::size_t, nZ, 1>> cstrSparsityVec(cstrSparsity.data());
    osqp_instance.objective_matrix.reserve(costSparsityVec);
    osqp_instance.constraint_matrix.reserve(cstrSparsityVec);

    // Get inital state
    dltv_pb_.get_x0(sol_.x.col(0));


    /* -------------------------------------------------------------------------- */
    /*                         Set state and input bounds                         */
    /* -------------------------------------------------------------------------- */

    for (std::size_t i = 0; i < nPts - 1; i++) {
      // state bounds
      dltv_pb_.get_state_lb(i + 1, osqp_instance.lower_bounds.segment<nx>(i * nx));
      dltv_pb_.get_state_ub(i + 1, osqp_instance.upper_bounds.segment<nx>(i * nx));
      // input bounds
      dltv_pb_.get_input_ub(i, osqp_instance.upper_bounds.segment<nu>(nX + i * nu));
      dltv_pb_.get_input_lb(i, osqp_instance.lower_bounds.segment<nu>(nX + i * nu));
    }

    // Set equality constraints bounds A(0)*x(0)
    osqp_instance.lower_bounds.segment<nx>(nZ) = dltv_pb_.get_A(0) * sol_.x.col(0);

    if constexpr (has_E) {
      osqp_instance.lower_bounds.segment<nx>(nZ) += dltv_pb_.get_E(0);

      for (std::size_t i = 1; i < nPts - 1; i++) {
        osqp_instance.lower_bounds.segment<nx>(nZ + (i * nx)) = dltv_pb_.get_E(i);
      }
    } else {
      osqp_instance.lower_bounds.segment<nx * (nPts - 2)>(nZ + nx) =
        Eigen::Matrix<double, nx *(nPts - 2), 1>::Zero();
    }

    // Duplicate lower bound constraints onto upper bounds to make equalities.
    osqp_instance.upper_bounds.segment<nEqCstr>(nZ) =
      osqp_instance.lower_bounds.segment<nEqCstr>(nZ);


    /* -------------------------------------------------------------------------- */
    /*                            Set contraint matrix A                          */
    /* -------------------------------------------------------------------------- */

    // Set bound constraints matrix
    for (std::size_t i = 0; i < nZ; i++) {
      osqp_instance.constraint_matrix.insert(i, i) = 1.;
    }

    // Fill up Go_eq Bottom upper half: I_(nx+nu*nx+nu)
    for (std::size_t i = nZ; i < nCstr; i++) {
      osqp_instance.constraint_matrix.insert(i, i - nZ) = 1.;
    }

    // Fill up Go_eq Bottom Left subDiagonal: -A (note i = 1 starting point)
    for (std::size_t i = 1; i < nPts - 1; i++) {
      const std::size_t i_col = (i - 1) * nx;
      const std::size_t i_row = nZ + i * nx;
      const Ar_t A = dltv_pb_.get_A(i);
      for (std::size_t c = 0; c < nx; ++c) {
        for (std::size_t r = 0; r < nx; ++r) {
          osqp_instance.constraint_matrix.insert(i_row + r, i_col + c) = -A(r, c);
        }
      }
    }

    // Fill up Go_eq Bottom Right Block Diagonal: -B
    for (std::size_t i = 0; i < nPts - 1; i++) {
      const std::size_t i_col = nX + i * nu;
      const std::size_t i_row = nZ + i * nx;
      const Br_t B = dltv_pb_.get_B(i);
      for (std::size_t c = 0; c < nu; ++c) {
        for (std::size_t r = 0; r < nx; ++r) {
          osqp_instance.constraint_matrix.insert(i_row + r, i_col + c) = -B(r, c);
        }
      }
    }


    /* -------------------------------------------------------------------------- */
    /*                   Set objective Vector q' = [qx qu]'                       */
    /* -------------------------------------------------------------------------- */

    //  fill in q
    if constexpr (has_q) {
      for (std::size_t i = 0; i < (nPts - 2); i++) {
        osqp_instance.objective_vector.segment<nx>(i * nx) = dltv_pb_.get_q(i + 1);
      }
    } else {
      osqp_instance.objective_vector.segment<nx * (nPts - 2)>(0).setZero();
    }
    //  fill in qT
    if constexpr (has_qT) {
      osqp_instance.objective_vector.segment<nx>((nPts - 2) * nx) = dltv_pb_.get_qT();
    } else {
      osqp_instance.objective_vector.segment<nx>((nPts - 2) * nx).setZero();
    }

    //  fill in r
    if constexpr (has_r) {
      for (std::size_t i = 0; i < (nPts - 1); i++) {
        osqp_instance.objective_vector.segment<nu>(nX + (i * nu)) = dltv_pb_.get_r(i);
      }
    } else {
      osqp_instance.objective_vector.segment<nu * (nPts - 1)>(nX).setZero();
    }


    /* -------------------------------------------------------------------------- */
    /*                     Set Objective Matrix P = [Q/QT/R]                      */
    /* -------------------------------------------------------------------------- */

    // Fill up Diagonal: Q
    for (std::size_t i = 0; i < nPts - 2; i++) {
      const std::size_t i_col = i * nx;
      const std::size_t i_row = i * nx;
      const Qr_t Q = dltv_pb_.get_Q(i + 1);
      for (std::size_t c = 0; c < nx; ++c) {
        for (std::size_t r = 0; r < c; ++r) {
          osqp_instance.objective_matrix.insert(i_row + r, i_col + c) = (Q(r, c) + Q(c, r)) / 2;
        }
        osqp_instance.objective_matrix.insert(i_row + c, i_col + c) = Q(c, c);
      }
    }

    // Fill up Diagonal: QT
    const std::size_t i_col = (nPts - 2) * nx;
    const std::size_t i_row = (nPts - 2) * nx;
    const QTr_t QT = dltv_pb_.get_QT();
    for (std::size_t c = 0; c < nx; ++c) {
      for (std::size_t r = 0; r < c; ++r) {
        osqp_instance.objective_matrix.insert(i_row + r, i_col + c) = (QT(r, c) + QT(c, r)) / 2;
      }
      osqp_instance.objective_matrix.insert(i_row + c, i_col + c) = QT(c, c);
    }

    // Fill up Diagonal: R
    for (std::size_t i = 0; i < nPts - 1; i++) {
      const std::size_t i_col = nX + i * nu;
      const std::size_t i_row = nX + i * nu;
      const Rr_t R = dltv_pb_.get_R(i);
      for (std::size_t c = 0; c < nu; ++c) {
        for (std::size_t r = 0; r < c; ++r) {
          osqp_instance.objective_matrix.insert(i_row + r, i_col + c) = (R(r, c) + R(c, r)) / 2;
        }
        osqp_instance.objective_matrix.insert(i_row + c, i_col + c) = R(c, c);
      }
    }

    /* ------------------------ compress Sparse Matrices ------------------------ */
    osqp_instance.objective_matrix.makeCompressed();
    osqp_instance.constraint_matrix.makeCompressed();

    const auto status = osqp_solver_.Init(osqp_instance, prm_.osqp_settings, true);

    if (!status.ok()) {
      throw std::runtime_error("Osqp initialization failed.");
    }

    osqp_instance_ = std::move(osqp_instance);
  }

  /**
   * @brief Set initial guess for problem solution
   *
   * @param x nx x nPts matrix of state values
   * @param u nu x nPts matrix of input values
   */
  template<typename Derived1, typename Derived2>
  void set_ic(const Eigen::MatrixBase<Derived1> & x, const Eigen::MatrixBase<Derived2> & u)
  {
    static_assert(
      Derived1::RowsAtCompileTime == nx,
      "Number of states inconsistant with the problem.");
    static_assert(
      Derived2::RowsAtCompileTime == nu,
      "Number of inputs inconsistant with the problem.");
    static_assert(
      Derived1::ColsAtCompileTime == nPts,
      "Number of state initial conditions inconsistent with the problem.");
    static_assert(
      Derived2::ColsAtCompileTime == nPts,
      "Number of input initial conditions inconsistent with the problem.");

    Eigen::Map<Eigen::Matrix<double, nx, nPts - 1>> varX(sol_.z.data());
    varX = x.template rightCols<nPts - 1>();

    Eigen::Map<Eigen::Matrix<double, nu, nPts - 1>> varU(sol_.z.data() + nX);
    varU = u.template leftCols<nPts - 1>();

    osqp_solver_.SetPrimalWarmStart(sol_.z);
  }

  /**
   * @brief Update solver parameters
   */
  template<typename T>
  void update_params(T && p)
  {
    prm_ = std::forward<T>(p);

    const auto status = osqp_solver_.Init(osqp_instance_, prm_.osqp_settings, true);

    if (!status.ok()) {
      throw std::runtime_error("Osqp initialization failed.");
    }
  }

  /**
   * @brief Solve the problem and return solution
   *
   * Note that init() must have been called prior to this function
   */
  Solution solve()
  {
    osqp::OsqpExitCode exit_code = osqp_solver_.Solve();

    if (exit_code == osqp::OsqpExitCode::kOptimal) {
      sol_.rc = DltvOcpSolverCode::success;
    } else {
      sol_.rc = DltvOcpSolverCode::failure;
    }

    sol_.z = osqp_solver_.primal_solution();

    // Create X
    const Eigen::Map<const Eigen::Matrix<double, nx, nPts - 1>> mX(sol_.z.data());
    sol_.x.template rightCols<nPts - 1>() = mX;

    // Create U
    const Eigen::Map<const Eigen::Matrix<double, nu, nPts - 1>> mU(sol_.z.data() + nX);

    sol_.u col(nPts - 1) = mU.template rightCols<1>();
    sol_.u.template leftCols<nPts - 1>() = mU;

    return sol_;
  }

  Solution solution() const
  {
    return sol_;
  }

  dltv_pb_t & problem()
  {
    return dltv_pb_;
  }

protected:
  dltv_pb_t dltv_pb_{};
  DltvOcpSolverParams prm_{};
  Solution sol_{};
  osqp::OsqpInstance osqp_instance_{};
  osqp::OsqpSolver osqp_solver_{};
};

// Class template argument deduction guides
template<typename T>
DltvOcpSolver(T)->DltvOcpSolver<T>;

template<typename T1, typename T2>
DltvOcpSolver(T1, T2)->DltvOcpSolver<T1>;

}  // namespace cbr

#endif  // CBR_CONTROL__MPC__DLTV_OCP_SOLVER_HPP_

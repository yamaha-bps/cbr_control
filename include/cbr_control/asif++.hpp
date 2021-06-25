// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__ASIF___HPP_
#define CBR_CONTROL__ASIF___HPP_

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <boost/numeric/odeint.hpp>
#include <boost/hana/adapt_struct.hpp>

#include <cbr_math/lie/odeint.hpp>
#include <cbr_math/lie/common.hpp>
#include <cbr_math/math.hpp>

#include <vector>

#include "cbr_control/osqp-cpp.hpp"


using autodiff::forward::dual, autodiff::forward::jacobian,
autodiff::forward::wrt, autodiff::forward::at;

using Eigen::Matrix;


namespace cbr
{

struct ASIFParams
{
  // integration time step and number of steps
  double dt{1};
  std::size_t steps{100};

  // spacing between added constraints (1 adds a constraint for each time step)
  std::size_t constr_dist{1};

  // cost multiplier of bounds relaxation
  double relax_cost{100};

  // barrier function constant
  double alpha{1};

  // solver tolerances
  double osqp_eps_abs{1e-4};
  double osqp_eps_rel{1e-4};
  bool osqp_polish{false};
  double osqp_timelimit{1e-2};
  std::size_t osqp_maxiter{5000};

  bool debug{false};
};


/**
 * @brief Active Set Invariance Filter
 *
 * @tparam Dyn: type with following functions
 *  Matrix<T, nx, 1> f(const State<T> &) const;
 *  Matrix<T, nx, nu> g(const State<T> &) const;
 * @tparam BackupContr: type with function
 *  Input<T> operator()(const State<T> &) const;
 * @tparam SafetySet: type with function
 *  Matrix<T, nh, 1> operator()(cosnt State<T> &) const;
 */
template<typename Dyn, typename BackupContr, typename SafetySet>
class ASIF
{
public:
  template<typename T>
  using State = typename Dyn::template State<T>;

  template<typename T>
  using Input = typename Dyn::template Input<T>;

  static constexpr std::size_t nx = State<double>::DoF;
  static constexpr std::size_t nu = Input<double>::SizeAtCompileTime;
  static constexpr std::size_t nh = decltype(SafetySet{} (State<double>{}))::RowsAtCompileTime;

  /**
   * @brief Construct a new ASIF
   *
   * @param dyn dynamics
   * @param bc backup controller
   * @param ss safety set described by level sets
   * @param params parameters
   */
  ASIF(const Dyn & dyn, const BackupContr & bc, const SafetySet & ss, const ASIFParams & params)
  : dyn_(dyn),
    bc_(bc),
    ss_(ss),
    prm_(params)
  {
    const std::size_t num_constr = prm_.steps / prm_.constr_dist;

    // QP constraints qp_l <= qp_A [u ; eps] <= qp_u

    // qp_A = [ B  1
    //          I  0 ]
    //
    // where B are barrier constraints

    // upper part of constraint matrix is dense, lower part has one nonzero per line
    Eigen::Matrix<std::size_t, -1, 1> qp_spA_sparsity(nu + num_constr * nh, 1);
    qp_spA_sparsity.head(nu + num_constr * nh).setConstant(nu + 1);
    qp_spA_sparsity.tail(nu).setOnes();

    qp_objvec_.setZero();

    qp_l_.resize(num_constr * nh + nu);
    qp_u_.resize(num_constr * nh + nu);

    qp_l_.setConstant(-OSQP_INFTY);
    qp_u_.setConstant(OSQP_INFTY);

    qp_A_.resize(num_constr * nh + nu, nu + 1);
    qp_A_.reserve(qp_spA_sparsity);

    for (auto i = 0u; i != num_constr * nh; ++i) {
      for (auto k = 0u; k != nu + 1; ++k) {
        qp_A_.insert(i, k) = 1;
      }
    }
    for (auto i = 0u; i != nu; ++i) {
      qp_A_.insert(num_constr * nh + i, i) = 1;
    }

    qp_A_.makeCompressed();

    Eigen::Matrix<std::size_t, -1, 1> objective_sparsity(nu + 1, 1);
    objective_sparsity.setOnes();

    osqp::OsqpInstance instance;

    // objective matrix does not change
    instance.objective_matrix.resize(nu + 1, nu + 1);
    instance.objective_matrix.reserve(objective_sparsity);
    for (auto i = 0u; i != nu; ++i) {instance.objective_matrix.insert(i, i) = 1;}
    instance.objective_matrix.insert(nu, nu) = prm_.relax_cost;
    instance.objective_matrix.makeCompressed();

    // objective vector changes
    instance.objective_vector = qp_objvec_;

    // constraint matrix changes
    instance.constraint_matrix = qp_A_;

    // constraint bounds change
    instance.lower_bounds = qp_l_;
    instance.upper_bounds = qp_u_;

    osqp::OsqpSettings settings;
    settings.verbose = prm_.debug;
    settings.eps_abs = prm_.osqp_eps_abs;
    settings.eps_rel = prm_.osqp_eps_rel;
    settings.polish = prm_.osqp_polish;
    settings.max_iter = prm_.osqp_maxiter;
    settings.time_limit = prm_.osqp_timelimit;

    osqp_solver_.Init(instance, settings, false);
  }

  /**
   * @brief Add input bounds to ASIF
   *
   * @param lb lower bounds
   * @param ub upper bounds
   */
  void setBounds(const Input<double> & lb, const Input<double> & ub)
  {
    qp_l_.template tail<nu>() = lb;
    qp_u_.template tail<nu>() = ub;
  }

  /**
   * @brief Filter an input with ASIF
   *
   * @param x[in] current state
   * @param u[in, out] desired input is modified in-place to satisfy barrier constraint
   * @param backup_traj[out] ouput backup trajectory (value for each QP constraint)
   * @param min_h[out] smallest h values along backup trajectory
   *
   * @return solver return code
   */
  osqp::OsqpExitCode filter(
    const State<double> & x,
    Input<double> & u,
    std::optional<
      std::reference_wrapper<
        std::vector<
          State<double>,
          Eigen::aligned_allocator<State<double>>
        >
      >
    > backup_traj = {},
    std::optional<Eigen::Ref<Eigen::Matrix<double, nh, 1>>> min_h = {}
  )
  {
    // initial f and g
    const Eigen::Matrix<double, nx, 1> f0 = dyn_.f(x);
    const Matrix<double, nx, nu> g0 = dyn_.g(x);

    // AD variable
    Matrix<dual, nx, 1> dx_ad = Matrix<dual, nx, 1>::Zero();

    // loop variables
    double ti = 0;
    State<double> xi = x;
    Matrix<double, nx, nx> dxi_dx0 = Matrix<double, nx, nx>::Identity();

    if (backup_traj.has_value()) {
      backup_traj.value().get().resize(prm_.steps / prm_.constr_dist);
    }

    for (auto i = 0u, j = 0u; i != prm_.steps; ++i, j = (j + 1) % prm_.constr_dist) {
      const auto xi_ad = xi.template cast<dual>();

      if (j == 0) {
        // evaluate h(x) and its derivative
        Matrix<dual, nh, 1> h_ad;
        const Matrix<double, nh, nx> dh_dxi = jacobian(
          [&](const auto & var) -> Matrix<dual, nh, 1> {return ss_(xi_ad * State<dual>::exp(var));},
          wrt(dx_ad), at(dx_ad), h_ad
        );
        const Matrix<double, nh, 1> h = h_ad.template cast<double>();

        // insert asif constraint in A and l
        const Matrix<double, nh, nx> dh_dx0 = dh_dxi * dxi_dx0;
        const Matrix<double, nh, nu> A_block = dh_dx0 * g0;

        if (prm_.debug) {
          std::cout << "dh_dxi" << std::endl << dh_dxi << std::endl;
          std::cout << "dxi_dx0" << std::endl << dxi_dx0 << std::endl;
          std::cout << "dh_dx0" << std::endl << dh_dx0 << std::endl;
          std::cout << "A_block" << std::endl << A_block << std::endl;
        }

        for (auto c = 0u; c != nu; ++c) {
          for (auto r = 0u; r != nh; ++r) {
            qp_A_.coeffRef((i / prm_.constr_dist) * nh + r, c) = A_block(r, c);
          }
        }
        qp_l_.template segment<nh>((i / prm_.constr_dist) * nh) = -prm_.alpha * h - dh_dx0 * f0;

        // store in backup trajectory
        if (backup_traj.has_value()) {
          backup_traj.value().get()[i / prm_.constr_dist] = xi;
        }

        if (min_h.has_value()) {
          min_h.value() = min_h.value().cwiseMin(h);
        }
      }


      // evaluate closed-loop dynamics fcl(x) and its derivative
      Matrix<dual, nx, 1> fcl_ad;
      const Matrix<double, nx, nx> dfcl_dxi = jacobian(
        [&](const auto & var) -> Matrix<dual, nx, 1> {
          const State<dual> x_dx = xi_ad * State<dual>::exp(var);
          const Matrix<dual, nu, 1> u = bc_(x_dx);
          const Matrix<dual, nu, 1> u_smooth = Matrix<dual, nu, 1>::NullaryExpr(
            [&](int i) {return smoothSat(u(i), qp_l_.tail<nu>()(i), qp_u_.tail<nu>()(i));}
          );
          return dyn_.f(x_dx) + dyn_.g(x_dx) * u_smooth;
        },
        wrt(dx_ad), at(dx_ad), fcl_ad
      );
      const Matrix<double, nx, 1> fcl = fcl_ad.template cast<double>();

      if (prm_.debug) {
        std::cout << "dfcl_dxi" << std::endl << dfcl_dxi << std::endl;
      }

      // NOTE should use ad() with lie library that supports it
      Matrix<double, nx, nx> adjoint_fcl;
      for (auto i = 0u; i != nx; ++i) {
        adjoint_fcl.col(i) = State<double>::lieBracket(fcl, Eigen::Matrix<double, nx, 1>::Unit(i));
      }

      // integrate system and sensitivity forward
      state_stepper.do_step(
        [&fcl](const State<double> &, Matrix<double, nx, 1> & dx_dt, double) {dx_dt = fcl;},
        xi, ti, prm_.dt
      );
      sensitivity_stepper.do_step(
        [&adjoint_fcl, &dfcl_dxi](const Matrix<double, nx, nx> & Dx,
        Matrix<double, nx, nx> & dDx_dt, double) {
          dDx_dt = (-adjoint_fcl + dfcl_dxi) * Dx;
        },
        dxi_dx0, ti, prm_.dt
      );

      ti += prm_.dt;
    }

    qp_objvec_.template head<nu>() = -u;

    if (prm_.debug) {
      std::cout << "Number of constraints " << qp_A_.rows() << std::endl;
      std::cout << "Setting constraint matrix to" << std::endl << qp_A_.toDense() << std::endl;
      std::cout << "Setting lb = " << qp_l_.transpose() << std::endl;
      std::cout << "Setting ub = " << qp_u_.transpose() << std::endl;
      std::cout << "Setting obj_vec = " << qp_objvec_.transpose() << std::endl;
    }

    osqp_solver_.UpdateConstraintMatrix(qp_A_);
    osqp_solver_.SetBounds(qp_l_, qp_u_);
    osqp_solver_.SetObjectiveVector(qp_objvec_);

    const auto rc = osqp_solver_.Solve();

    if (rc == osqp::OsqpExitCode::kOptimal || rc == osqp::OsqpExitCode::kOptimalInaccurate) {
      // return optimal solution
      u = osqp_solver_.primal_solution().template head<nu>();
    } else {
      // return saturated backup trajectory if solving failed
      const Matrix<double, nu, 1> u_backup = bc_(x);
      u = Matrix<double, nu, 1>::NullaryExpr(
        [&](int i) {return smoothSat(u_backup(i), qp_l_.tail<nu>()(i), qp_u_.tail<nu>()(i));}
      );
    }

    return rc;
  }

private:
  // problem definition
  Dyn dyn_;
  BackupContr bc_;
  SafetySet ss_;
  ASIFParams prm_;

  // steppers
  cbr::lie::odeint::euler<State<double>, double, Eigen::Matrix<double, nx, 1>> state_stepper{};
  boost::numeric::odeint::euler<
    Matrix<double, nx, nx>, double, Matrix<double, nx, nx>, double,
    boost::numeric::odeint::vector_space_algebra
  > sensitivity_stepper{};

  // qp matrices
  Eigen::SparseMatrix<double, Eigen::RowMajor> qp_A_;
  Eigen::Matrix<double, -1, 1> qp_l_, qp_u_;
  Eigen::Matrix<double, nu + 1, 1> qp_objvec_;

  // qp solver
  osqp::OsqpSolver osqp_solver_;
};

}  // namespace cbr

// cppcheck-suppres unknownMacro
BOOST_HANA_ADAPT_STRUCT(
  cbr::ASIFParams,
  dt,
  steps,
  constr_dist,
  relax_cost,
  alpha,
  osqp_eps_abs,
  osqp_eps_rel,
  osqp_polish,
  osqp_timelimit,
  osqp_maxiter,
  debug
);

#endif  // CBR_CONTROL__ASIF___HPP_

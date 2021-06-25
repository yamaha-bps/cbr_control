// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__MPC_TRACKING_HPP_
#define CBR_CONTROL__MPC__MPC_TRACKING_HPP_

#include <Eigen/Dense>
#include <cbr_utils/thread_pool.hpp>
#include <cbr_utils/utils.hpp>
#include <cbr_math/interp.hpp>

#include <utility>
#include <future>
#include <mutex>
#include <thread>

#include "cltv_ocp.hpp"
#include "cltv_ocp_lie.hpp"
#include "dltv_ocp.hpp"
#include "dltv_ocp_solver.hpp"
#include "ocp_common.hpp"

using std::chrono::duration_cast, std::chrono::duration, std::chrono::nanoseconds;

namespace cbr
{

/**
 * @brief Helper type for MPC trajectory tracking
 *
 * CltvOcp requires an interface with get_xxx(tau) methods where tau is
 * the time elapsed from the initial condition of the problem.
 *
 * This class takes functions of absolute time t and implements the required
 * interfaces as functions of relative time.
 */
template<typename _problem_t>
struct MPCTrackingProblem : public _problem_t
{
public:
  using typename _problem_t::state_t;
  using typename _problem_t::deriv_t;
  using typename _problem_t::input_t;

  explicit MPCTrackingProblem(double T, const _problem_t & p)
  : _problem_t(p), T_(T) {}

  explicit MPCTrackingProblem(double T, _problem_t && p)
  : _problem_t(std::move(p)), T_(T) {}

  /**
   * @brief Return the time horizon of the problem
   * @param[out] T result
   */
  double get_T() const {return T_;}

  /**
   * @brief Return the problem initial condition
   * @param[out] x0 result
   */
  const state_t & get_x0() const {return x0;}

  /**
   * @brief Get the desired state
   * @param tau relative problem time tau = t - t0
   * @return xd(t0 + tau)
   */
  state_t get_xd(double tau) const
  {
    return xd(t0 + duration_cast<nanoseconds>(duration<double>(tau)));
  }
  /**
   * @brief Get the desired input
   * @param tau relative problem time tau = t - t0
   * @return ud(t0 + tau)
   */
  input_t get_ud(double tau) const
  {
    return ud(t0 + duration_cast<nanoseconds>(duration<double>(tau)));
  }

  /**
   * @brief Get the state linearization
   * @param tau relative problem time tau = t - t0
   * @return xl(t0 + tau)
   */
  state_t get_xl(double tau) const
  {
    return xl(t0 + duration_cast<nanoseconds>(duration<double>(tau)));
  }
  /**
   * @brief Get the state derivative linearization
   * @param tau relative problem time tau = t - t0
   * @return xldot(t0 + tau)
   */
  deriv_t get_xldot(double tau) const
  {
    return xldot(t0 + duration_cast<nanoseconds>(duration<double>(tau)));
  }
  /**
   * @brief Get the input linearization
   * @param tau relative problem time tau = t - t0
   * @return ul(t0 + tau)
   */
  input_t get_ul(double tau) const
  {
    return ul(t0 + duration_cast<nanoseconds>(duration<double>(tau)));
  }

private:
  // These functions return are defined in absolute time
  std::function<state_t(nanoseconds)> xd = [](nanoseconds) {return state_t{};};
  std::function<input_t(nanoseconds)> ud = [](nanoseconds) {return input_t{};};
  std::function<state_t(nanoseconds)> xl = [](nanoseconds) {return state_t{};};
  std::function<deriv_t(nanoseconds)> xldot = [](nanoseconds) {return deriv_t{};};
  std::function<input_t(nanoseconds)> ul = [](nanoseconds) {return input_t{};};

  nanoseconds t0 = nanoseconds(0);
  state_t x0{};
  double T_{0};

  // MPCTracking modifies the private variables from outside (except T_)
  template<typename problem_t__, std::size_t nPts__>
  friend class MPCTracking;
};


struct MPCTrackingParams
{
  double T{10};  // MPC horizon
  DltvOcpSolverParams solver_params{};
};

// Trait to detect whether a problem is defined on a lie group
template<typename, typename = void>
struct has_tangent : std::false_type {};

template<typename T>
struct has_tangent<T, std::void_t<typename T::Tangent>>: std::true_type {};


/**
 * @brief Trajectory tracking with nonlinear MPC
 *
 * @tparam problem_t stationary problem definition with dynamics, weights, and state bounds
 * @tparam nPts_ number of time discretization points
 *
 * NOTE: all set_xxx methods must be called before solving for the first time
 */
template<typename _problem_t, std::size_t nPts_>
class MPCTracking
{
public:
  // Linearization order used for system linearization
  static constexpr std::size_t LinOrder = 4;

  static constexpr bool is_lie = has_tangent<typename _problem_t::state_t>::value;

  using problem_t = _problem_t;
  using state_t = typename problem_t::state_t;
  using deriv_t = typename problem_t::deriv_t;
  using input_t = typename problem_t::input_t;  // must be eigen type

  // Use CltvOcpLie or CltvOcp depending on the type of problem we have
  using cltv_t = typename std::conditional<is_lie,
      CltvOcpLie<MPCTrackingProblem<_problem_t>>,
      CltvOcp<MPCTrackingProblem<_problem_t>>
    >::type;

  using dltv_t = DltvOcp<cltv_t, nPts_, LinOrder>;
  using solver_t = DltvOcpSolver<dltv_t>;

  using state_traj_t = typename std::conditional<is_lie,
      std::array<state_t, nPts_>,
      Eigen::Matrix<double, problem_t::nx, nPts_>
    >::type;

  struct Solution
  {
    DltvOcpSolverCode rc;
    Eigen::Matrix<double, 1, nPts_> t;
    state_traj_t x;
    Eigen::Matrix<double, input_t::SizeAtCompileTime, nPts_> u;
  };

  MPCTracking() = delete;
  MPCTracking(const MPCTracking &) = default;
  MPCTracking(MPCTracking &&) = default;
  MPCTracking & operator=(const MPCTracking &) = default;
  MPCTracking & operator=(MPCTracking &&) = default;

  explicit MPCTracking(const _problem_t & problem, MPCTrackingParams param = MPCTrackingParams{})
  : solver_(
      DltvOcp<cltv_t, nPts_, LinOrder>(cltv_t(MPCTrackingProblem<_problem_t>(param.T, problem))),
      param.solver_params
  ),
    uspline_(
      (Eigen::Matrix<double, 1, 2>() << 0., 1.).finished(),
      input_t::Zero()
    ),
    tp_(1)
  {
    mpcSol_.rc = DltvOcpSolverCode::no_run;

    lock_problem();

    if constexpr (is_lie) {
      // lie types default-initialize to identity
      set_xd([](std::chrono::nanoseconds) {return state_t{};});
      set_xl([](std::chrono::nanoseconds) {return state_t{};});
      set_xldot([](std::chrono::nanoseconds) {return deriv_t{};});
    } else {
      // Eigen types must be explicitly zero-initialized
      set_xd([](std::chrono::nanoseconds) {return state_t::Zero();});
      set_xl([](std::chrono::nanoseconds) {return state_t::Zero();});
      set_xldot([](std::chrono::nanoseconds) {return deriv_t::Zero();});
    }

    // zero-initialize linearization and desired inputs
    set_ud([](std::chrono::nanoseconds) {return input_t::Zero();});
    set_ul([](std::chrono::nanoseconds) {return input_t::Zero();});

    unlock_problem();
  }


  /**
   * @brief Block solving to safely update xd/ud/xl/xldot/ul
   *
   * Use before calling the set_xxx functions, then call unlock_problem()
   */
  void lock_problem() {problemsMtx_.lock();}


  /**
   * @brief Unblock solving
   *
   * Use after calling lock_problem()
   */
  void unlock_problem() {problemsMtx_.unlock();}


  /**
   * @brief Specify desired state trajectory as a function t -> xd(t) [absolute time]
   *
   * Only use after calling lock_problem()
   *
   * @param f xd as a function of t
   */
  template<typename T>
  void set_xd(T && f) {solver_.problem().problem().problem().xd = std::forward<T>(f);}


  /**
   * @brief Specify desired input trajectory as a function t -> ud(t) [absolute time]
   *
   * Only use after calling lock_problem()
   *
   * @param f ud as a function of t
   */

  template<typename T>
  void set_ud(T && f) {solver_.problem().problem().problem().ud = std::forward<T>(f);}


  /**
   * @brief Specify state linearization trajectory as a function t -> xd(t) [absolute time]
   *
   * Only use after calling lock_problem()
   *
   * May be required if xd/ud is changed to be far from current linearization
   */
  template<typename T>
  void set_xl(T && f) {solver_.problem().problem().problem().xl = std::forward<T>(f);}


  /**
   * @brief Specify state linearization trajectory as a function t -> \dot xd(t) [absolute time]
   *
   * Only use after calling lock_problem()
   *
   * May be required if xd/ud is changed to be far from current linearization
   */
  template<typename T>
  void set_xldot(T && f) {solver_.problem().problem().problem().xldot = std::forward<T>(f);}


  /**
   * @brief Specify state linearization trajectory as a function t -> xd(t) [absolute time]
   *
   * Only use after calling lock_problem()
   *
   * May be required if xd/ud is changed to be far from current linearization
   */
  template<typename T>
  void set_ul(T && f) {solver_.problem().problem().problem().ul = std::forward<T>(f);}


  /**
   * @brief Asynchronous update of MPC
   *
   * @param t current absolute time
   * @param xt state at time t
   * @return future to solver status
   *
   * If optimization is already running this function returns without without effect.
   */
  std::shared_future<DltvOcpSolverCode>
  update(const nanoseconds t, const state_t & xt)
  {
    std::lock_guard lock(futMtx_);
    if (fut_.valid()) {
      if (fut_.wait_for(nanoseconds(0)) != std::future_status::ready) {
        return {};
      }
    }

    fut_ = tp_.enqueue(&MPCTracking::update_, this, t, xt, 1);
    return fut_.share();
  }


  /**
   * @brief Blocking update of MPC
   *
   * @param t current absolute time
   * @param xt state at time t
   * @param iter number of solve/linearaze iterations to run
   * @return solver status
   */
  DltvOcpSolverCode update_sync(const nanoseconds t, const state_t & xt, std::size_t iter = 1)
  {
    std::lock_guard lock(futMtx_);
    if (fut_.valid()) {
      fut_.wait();
    }
    return update_(t, xt, iter);
  }


  /**
   * @brief Obtain input at given time for most recent solution
   */
  input_t get_u(nanoseconds t)
  {
    std::lock_guard lock(solutionMtx_);
    return uspline_.val(duration_cast<duration<double>>(t - uspline_t0_).count());
  }


  /**
   * @brief Obtain most recent solution
   *
   * NOTE: solution times are defined on [0, T]
   */
  Solution solution()
  {
    std::lock_guard lock(solutionMtx_);
    return mpcSol_;
  }

private:
  /**
   * @brief Internal helper function to update mpc solution
   */
  DltvOcpSolverCode update_(std::chrono::nanoseconds t0, const state_t & x0, std::size_t iter)
  {
    std::lock_guard lock(problemsMtx_);  // lock for the duration to ensure consistency
    DltvOcpSolverCode rc = DltvOcpSolverCode::no_run;

    solver_.problem().problem().problem().t0 = t0;
    solver_.problem().problem().problem().x0 = x0;

    for (std::size_t i = 0; i != iter; ++i) {
      solver_.init();

      auto sol = solver_.solve();
      rc = sol.rc;

      if (sol.rc != DltvOcpSolverCode::success) {
        break;
      } else {
        Solution abs_sol;
        abs_sol.rc = sol.rc;

        for (auto i = 0u; i < nPts_; i++) {
          // solution is a trajectory around the linearization, we add back
          // the linearization to get a trajectory in global coordinates
          const double tt = solver_.problem().indexToTime(i);
          abs_sol.t[i] = tt;
          if constexpr (is_lie) {
            abs_sol.x[i] =
              solver_.problem().problem().problem().get_xl(tt) * state_t::exp(sol.x.col(i));
          } else {
            abs_sol.x.col(i) = solver_.problem().problem().problem().get_xl(tt) + sol.x.col(i);
          }
          abs_sol.u.col(i) = solver_.problem().problem().problem().get_ul(tt) + sol.u.col(i);
        }

        auto ulin = cbr::PiecewiseLinear::fitND(abs_sol.t, abs_sol.u);

        {
          // store solution so that get_u() can be called
          std::lock_guard lock(solutionMtx_);
          mpcSol_ = abs_sol;

          uspline_t0_ = t0;
          uspline_ = ulin;
        }

        // Set desired input for future solving to current input
        solver_.problem().problem().problem().ud =
          [u0 = abs_sol.u.col(0).eval()](nanoseconds) -> input_t {
            return u0;
          };

        // Update linearization points (functions defined in absolute time)
        solver_.problem().problem().problem().ul =
          [t0 = t0, ulin = std::move(ulin)](nanoseconds t) -> input_t {
            return ulin.val(duration_cast<duration<double>>(t - t0).count());
          };

        if constexpr (is_lie) {
          // fit splines: first need to copy to a vector
          vector_aligned<state_t> sol_vec(abs_sol.x.begin(), abs_sol.x.end());
          auto xlin = cbr::Spline::fitLie(abs_sol.t, std::move(sol_vec));
          xlin.set_extrap(PiecewisePoly::EXTRAP::CLAMP);

          solver_.problem().problem().problem().xl =
            [t0 = t0, xlin = xlin](nanoseconds t) -> state_t {
              const auto t_spline = duration_cast<duration<double>>(t - t0).count();
              return xlin.val(t_spline);
            };
          solver_.problem().problem().problem().xldot =
            [t0 = t0, xlin = std::move(xlin)](nanoseconds t) -> deriv_t {
              const auto t_spline = duration_cast<duration<double>>(t - t0).count();
              return xlin.der(t_spline);
            };
        } else {
          // fit splines
          auto xlin = cbr::PiecewiseLinear::fitND(abs_sol.t, abs_sol.x);

          solver_.problem().problem().problem().xl =
            [t0 = t0, xlin = xlin](nanoseconds t) -> state_t {
              return xlin.val(duration_cast<duration<double>>(t - t0).count());
            };
          solver_.problem().problem().problem().xldot =
            [t0 = t0, xlin = xlin](nanoseconds t) -> deriv_t {
              return xlin.der(duration_cast<duration<double>>(t - t0).count());
            };
        }

        // reset warm-start solution to zeros
        solver_.set_ic(decltype(sol.x)::Zero(), decltype(sol.u)::Zero());
      }
    }

    return rc;
  }

private:
  solver_t solver_{};

  Solution mpcSol_;
  std::mutex solutionMtx_, problemsMtx_, futMtx_;
  std::future<DltvOcpSolverCode> fut_;

  nanoseconds uspline_t0_;
  cbr::PiecewisePolyND uspline_;

  ThreadPool tp_;
};

}  // namespace cbr

#endif  // CBR_CONTROL__MPC__MPC_TRACKING_HPP_

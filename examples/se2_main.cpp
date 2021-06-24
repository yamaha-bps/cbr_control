// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>
#include <cbr_math/lie/odeint.hpp>
#include <cbr_control/mpc/mpc_tracking.hpp>
#include <matplot/matplot.h>

#include <chrono>
#include <vector>
#include <algorithm>

#include "se2_problem.hpp"

using namespace std::chrono_literals;


int main(int argc, char const * argv[])
{
  using state_t = typename SE2Problem::state_t;
  using deriv_t = typename SE2Problem::deriv_t;
  using input_t = typename SE2Problem::input_t;

  //  -------------------------------------------------------------------------- /
  //                              Simulation Params                              /
  //  -------------------------------------------------------------------------- /

  state_t x0{};
  const auto tf = 5s;
  const auto dt = 10ms;

  auto xd = [](nanoseconds) {
      return state_t(
        Sophus::SE2d(Sophus::SO2d(0.25), Sophus::Vector2d(0.5, 1)),
        Eigen::Vector3d::Zero()
      );
    };

  //  -------------------------------------------------------------------------- /
  //                                      MPC                                    /
  //  -------------------------------------------------------------------------- /

  SE2Problem ocp{};

  cbr::MPCTrackingParams params;
  params.T = 4;
  params.solver_params.osqp_settings.verbose = 0;
  params.solver_params.osqp_settings.polish = 1;
  params.solver_params.osqp_settings.eps_abs = 1e-5;
  params.solver_params.osqp_settings.eps_rel = 1e-5;

  cbr::MPCTracking<SE2Problem, 50> mpc(ocp, params);
  mpc.set_xd(xd);

  //  -------------------------------------------------------------------------- /
  //                                RUN SIMULATION                               /
  //  -------------------------------------------------------------------------- /

  nanoseconds t(0);
  state_t x = x0;

  cbr::lie::odeint::runge_kutta4<state_t, double, deriv_t, double> stepper;

  std::vector<double> sol_t;
  std::vector<input_t, Eigen::aligned_allocator<input_t>> sol_u;
  std::vector<state_t, Eigen::aligned_allocator<state_t>> sol_x;

  while (t < tf) {
    mpc.update_sync(t, x);
    const auto u = mpc.get_u(t);

    sol_t.push_back(duration_cast<duration<double>>(t).count());
    sol_x.push_back(x);
    sol_u.push_back(u);

    stepper.do_step(
      [&ocp, &u](const state_t & x, deriv_t & dr_x, const double) {
        dr_x = ocp.get_f(x, u);
      },
      x,
      duration_cast<duration<double>>(t).count(),
      duration_cast<duration<double>>(dt).count()
    );

    t += dt;
  }

  //  -------------------------------------------------------------------------- /
  //                                PLOT RESULTS                                 /
  //  -------------------------------------------------------------------------- /

  // helper function to extract stuff from solutions
  auto ex_fn = [](const auto & item, auto ex_fn) {
      std::vector<double> ret;
      std::transform(item.cbegin(), item.cend(), std::back_inserter(ret), ex_fn);
      return ret;
    };


  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<0>(s).translation().x();}))->line_width(2);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<0>(s).translation().y();}))->line_width(2);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<0>(s).so2().log();}))->line_width(2);
  matplot::title("angles");
  matplot::legend({"x", "y", "yaw"});
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<1>(s).translation()(0);}))->line_width(2);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<1>(s).translation()(1);}))->line_width(2);
  matplot::plot(
    sol_t,
    ex_fn(sol_x, [](auto s) {return std::get<1>(s).translation()(2);}))->line_width(2);
  matplot::title("velocities");
  matplot::legend({"vx", "vy", "wz"});
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(0);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(1);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(2);}))->line_width(2);
  matplot::title("inputs");
  matplot::legend({"ux", "uy", "uw"});
  matplot::show();

  return EXIT_SUCCESS;
}

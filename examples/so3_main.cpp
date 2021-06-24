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

#include "so3_problem.hpp"

using namespace std::chrono_literals;


int main(int argc, char const * argv[])
{
  using state_t = SO3Problem::state_t;
  using deriv_t = SO3Problem::deriv_t;
  using input_t = SO3Problem::input_t;

  //  -------------------------------------------------------------------------- /
  //                              Simulation Params                              /
  //  -------------------------------------------------------------------------- /

  state_t x0{};
  const auto tf = 10s;
  const auto dt = 10ms;

  auto xd = [](nanoseconds) {
      return state_t(
        Sophus::SO3d::rotZ(0.25) * Sophus::SO3d::rotY(-0.4) * Sophus::SO3d::rotX(0.5),
        Eigen::Vector3d::Zero()
      );
    };

  //  -------------------------------------------------------------------------- /
  //                                      MPC                                    /
  //  -------------------------------------------------------------------------- /

  SO3Problem so3_ocp{};

  cbr::MPCTrackingParams params;
  params.T = 4;
  params.solver_params.osqp_settings.verbose = 1;

  cbr::MPCTracking<SO3Problem, 50> mpc(so3_ocp, params);
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
      [&so3_ocp, &u](const state_t & x, deriv_t & dr_x, const double) {
        dr_x = so3_ocp.get_f(x, u);
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
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return std::get<0>(s).angleX();}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return std::get<0>(s).angleY();}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return std::get<0>(s).angleZ();}))->line_width(2);
  matplot::title("angles");
  matplot::legend({"roll", "pitch", "yaw"});
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
  matplot::legend({"vx", "vy", "vz"});
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(0);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(1);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(2);}))->line_width(2);
  matplot::title("inputs");
  matplot::legend({"ux", "uy", "uz"});
  matplot::show();

  return EXIT_SUCCESS;
}

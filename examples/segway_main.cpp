// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>
#include <cbr_math/lie/odeint.hpp>
#include <cbr_control/mpc/mpc_tracking.hpp>
#include <matplot/matplot.h>

#include <chrono>
#include <vector>
#include <algorithm>

#include "segway_problem.hpp"

using namespace std::chrono_literals;


int main(int argc, char const * argv[])
{
  using state_t = SegwayProblem::state_t;
  using deriv_t = SegwayProblem::deriv_t;
  using input_t = SegwayProblem::input_t;

  //  -------------------------------------------------------------------------- /
  //                              Simulation Params                              /
  //  -------------------------------------------------------------------------- /

  state_t x0 = state_t::Zero();
  const auto tf = 10s;
  const auto dt = 10ms;

  auto xd = [](nanoseconds) {
      return state_t::UnitX();
    };

  //  -------------------------------------------------------------------------- /
  //                                      MPC                                    /
  //  -------------------------------------------------------------------------- /

  SegwayProblem ocp{};

  cbr::MPCTrackingParams params;
  params.T = 6;
  params.solver_params.osqp_settings.verbose = 1;

  cbr::MPCTracking<SegwayProblem, 101> mpc(ocp, params);
  mpc.set_xd(xd);

  //  -------------------------------------------------------------------------- /
  //                                RUN SIMULATION                               /
  //  -------------------------------------------------------------------------- /

  nanoseconds t(0);
  state_t x = x0;

  boost::numeric::odeint::euler<
    state_t, double, state_t, double,
    boost::numeric::odeint::vector_space_algebra
  > stepper;

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
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return s(0);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return s(1);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return s(2);}))->line_width(2);
  matplot::title("pose");
  matplot::legend({"x", "y", "psi"});
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(0);}))->line_width(2);
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(1);}))->line_width(2);
  matplot::title("inputs");
  matplot::legend({"u1", "u2"});
  matplot::show();

  return EXIT_SUCCESS;
}

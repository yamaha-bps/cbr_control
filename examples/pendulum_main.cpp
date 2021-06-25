// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <Eigen/Dense>

#include <boost/numeric/odeint.hpp>
#include <cbr_control/mpc/mpc_tracking.hpp>
#include <matplot/matplot.h>

#include <chrono>
#include <vector>
#include <algorithm>

#include "pendulum_problem.hpp"

using namespace std::chrono_literals;


int main(int argc, char const * argv[])
{
  using state_t = Eigen::Matrix<double, 2, 1>;  // [ theta, theta_dot ]
  using input_t = Eigen::Matrix<double, 1, 1>;  // [ tau ]

  //  -------------------------------------------------------------------------- /
  //                              Simulation Params                              /
  //  -------------------------------------------------------------------------- /

  state_t x0 = (state_t() << cbr::deg2rad(5.), cbr::deg2rad(0.)).finished();
  const auto dt = 10ms;
  const auto tf = 5s;

  auto xd = [](nanoseconds) {
      return state_t(cbr::deg2rad(0.), 0);
    };

  //  -------------------------------------------------------------------------- /
  //                                      MPC                                    /
  //  -------------------------------------------------------------------------- /

  NlOcpPendulum nlOcp{};

  cbr::MPCTrackingParams params;
  params.T = 5;
  params.solver_params.osqp_settings.verbose = 0;

  cbr::MPCTracking<NlOcpPendulum, 101> mpc(nlOcp, params);
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
  std::vector<state_t, Eigen::aligned_allocator<state_t>> sol_x;
  std::vector<input_t, Eigen::aligned_allocator<input_t>> sol_u;

  while (t < tf) {
    mpc.update_sync(t, x);
    const auto u = mpc.get_u(t);

    sol_t.push_back(duration_cast<duration<double>>(t).count());
    sol_x.push_back(x);
    sol_u.push_back(u);

    stepper.do_step(
      [&nlOcp, &u](const state_t & x, state_t & xdot, const double) {
        xdot = nlOcp.get_f(x, u);
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
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return s(0);}))->line_width(2);
  matplot::title("theta");
  matplot::figure();
  matplot::plot(sol_t, ex_fn(sol_x, [](auto s) {return s(1);}))->line_width(2);
  matplot::title("dot theta");
  matplot::figure();
  matplot::plot(sol_t, ex_fn(sol_u, [](auto s) {return s(0);}))->line_width(2);
  matplot::title("input");
  matplot::show();

  return EXIT_SUCCESS;
}

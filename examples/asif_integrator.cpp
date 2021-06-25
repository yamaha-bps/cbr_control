// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <boost/numeric/odeint.hpp>

#include <cbr_math/lie/group_product.hpp>
#include <cbr_math/lie/Tn.hpp>

#include <cbr_control/asif++.hpp>

#include <matplot/matplot.h>

#include <algorithm>
#include <vector>

struct Dynamics
{
  template<typename T>
  using State = cbr::lie::T2<T>;

  template<typename T>
  using Input = Eigen::Matrix<T, 1, 1>;

  template<typename T>
  Eigen::Matrix<T, 2, 1> f(const State<T> & x) const
  {
    return Eigen::Matrix<T, 2, 1>(x.translation()(1), 0);
  }

  template<typename T>
  Eigen::Matrix<T, 2, 1> g(const State<T> & x) const
  {
    return Eigen::Matrix<T, 2, 1>(0, 1);
  }
};

struct SS
{
  template<typename T>
  Eigen::Matrix<T, 2, 1> operator()(const cbr::lie::T2<T> & x) const
  {
    return Eigen::Matrix<T, 2, 1>(
      T(2) - x.translation().x(),
      T(2) - x.translation().y()
    );
  }
};

struct Backup
{
  template<typename T>
  Eigen::Matrix<T, 1, 1> operator()(const Dynamics::State<T> & x) const
  {
    return Eigen::Matrix<T, 1, 1>(-0.6);
  }
};


int main()
{
  cbr::ASIFParams params;
  params.dt = 0.02;
  params.steps = 200;
  params.constr_dist = 10;
  params.relax_cost = 500;
  params.debug = false;

  Dynamics dyn;
  Backup bu;
  SS ss;

  cbr::ASIF asif(dyn, bu, ss, params);
  asif.setBounds(Eigen::Matrix<double, 1, 1>(-0.5), Eigen::Matrix<double, 1, 1>(1.5));

  std::vector<double> sol_t;
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> sol_x;
  std::vector<double> sol_u, sol_udes;

  Eigen::Vector2d x(-5, 1);
  double udes = 1;

  double t = 0;
  double dt = 0.01;

  boost::numeric::odeint::runge_kutta4<Eigen::Vector2d, double, Eigen::Vector2d, double,
    boost::numeric::odeint::vector_space_algebra> stepper {};

  for (int i = 0; i != static_cast<int>(10. / dt); ++i) {
    Eigen::Matrix<double, 1, 1> u(udes);
    asif.filter(cbr::lie::T2d(x), u);

    sol_t.push_back(t);
    sol_x.push_back(x);
    sol_udes.push_back(udes);
    sol_u.push_back(u(0));

    stepper.do_step(
      [&](const Eigen::Vector2d & x, Eigen::Vector2d & dx, double t) {dx(0) = x(1); dx(1) = u(0);},
      x, t, dt
    );
    t += dt;
  }

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
  matplot::title("states");
  matplot::legend({"x", "v"});
  matplot::figure();
  matplot::hold(matplot::on);
  matplot::plot(sol_t, sol_udes)->line_width(2);
  matplot::plot(sol_t, sol_u)->line_width(2);
  matplot::title("input");
  matplot::legend({"u_{des}", "u"});

  matplot::show();

  return EXIT_SUCCESS;
}

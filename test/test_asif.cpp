// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#include <gtest/gtest.h>

#include <cbr_math/lie/group_product.hpp>
#include <cbr_math/lie/Tn.hpp>

#include "cbr_control/asif++.hpp"

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
  Eigen::Matrix<T, 2, 1> g(const State<T> &) const
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
  Eigen::Matrix<T, 1, 1> operator()(const Dynamics::State<T> &) const
  {
    return Eigen::Matrix<T, 1, 1>(-0.6);
  }
};

template class cbr::ASIF<Dynamics, Backup, SS>;

TEST(ASIF, UpperBound)
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
  asif.setBounds(Eigen::Matrix<double, 1, 1>(-1), Eigen::Matrix<double, 1, 1>(1));

  cbr::lie::T2d x(Eigen::Matrix<double, 2, 1>(-2, 1));
  Eigen::Matrix<double, 1, 1> u(2);

  asif.filter(x, u);

  ASSERT_LE(u.x(), 1 + 1e-5);
}


TEST(ASIF, IntegratorAvoid)
{
  cbr::ASIFParams params;
  params.dt = 0.02;
  params.steps = 200;
  params.constr_dist = 4;
  params.alpha = 0.5;
  params.relax_cost = 5000;
  params.debug = false;

  Dynamics dyn;
  Backup bu;
  SS ss;

  cbr::ASIF asif(dyn, bu, ss, params);
  asif.setBounds(Eigen::Matrix<double, 1, 1>(-0.5), Eigen::Matrix<double, 1, 1>(1.5));

  Eigen::Vector2d x(-5, 1);
  double udes = 1;

  double t = 0;
  double dt = 0.01;

  boost::numeric::odeint::runge_kutta4<Eigen::Vector2d, double, Eigen::Vector2d, double,
    boost::numeric::odeint::vector_space_algebra> stepper {};

  for (int i = 0; i != static_cast<int>(10. / dt); ++i) {
    Eigen::Matrix<double, 1, 1> u(udes);
    asif.filter(cbr::lie::T2d(x), u);

    ASSERT_GE(ss(cbr::lie::T2d(x)).minCoeff(), 0);

    stepper.do_step(
      [&](const Eigen::Vector2d & x, Eigen::Vector2d & dx, double) {dx(0) = x(1); dx(1) = u(0);},
      x, t, dt
    );
    t += dt;
  }
}

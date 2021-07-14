// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <gtest/gtest.h>

#include <boost/numeric/odeint.hpp>

#include <cbr_math/lie/odeint.hpp>
#include <cbr_control/pge/pge.hpp>

#include <memory>
#include <tuple>
#include <utility>


template<typename T>
using state_t = cbr::lie::GroupProduct<T, 0,
    Sophus::SE2,  // pose
    cbr::lie::T3  // body vel
>;
template<typename T>
using dstate_t = typename state_t<T>::Tangent;

using input_t = Eigen::Vector2d;


struct Dynamics
{
  static constexpr int DoF = state_t<double>::DoF;
  using sqrt_inf_t = Eigen::Matrix<double, DoF, DoF>;

  template<typename Derived1, typename Derived2>
  typename cbr::lie::GroupProductBase<Derived1>::Tangent
  operator()(
    const cbr::lie::GroupProductBase<Derived1> & x,
    const Eigen::MatrixBase<Derived2> & u) const
  {
    using Scalar = typename Derived1::Scalar;

    typename state_t<Scalar>::Tangent res;
    res.setZero();

    res.template head<3>() = std::get<1>(static_cast<const Derived1 &>(x)).translation();
    res(3) = Scalar(u[0]);  // forward acceleration
    res(5) = Scalar(u[1]);  // angular acceleration

    return res;
  }

  template<typename T>
  sqrt_inf_t sqrt_inf(const state_t<T> &) const
  {
    return state_t<double>::Tangent::Ones().asDiagonal();
  }
};


struct PosRes
{
  PosRes(Sophus::Vector2d meas, Sophus::Vector2d sqrt_inf)
  : meas_(meas), sqrt_inf_(sqrt_inf)
  {}

  template<typename T>
  Eigen::Matrix<T, 2, 1> operator()(Eigen::Map<const state_t<T>> x) const
  {
    Eigen::Matrix<T, 2, 1> res = std::get<0>(x).translation() - meas_;
    res.applyOnTheLeft(sqrt_inf_.template cast<T>().asDiagonal());
    return res;
  }

  Sophus::Vector2d meas_;
  Sophus::Vector2d sqrt_inf_;  // square root information: 1./stdev

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


using DubinsPGE = cbr::PoseGraphEstimator<state_t, input_t, Dynamics, PosRes>;

TEST(PGE, DubinsExample)
{
  spdlog::set_level(spdlog::level::off);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution;
  auto gauss = [&](int) {return distribution(generator);};

  constexpr int kStateSize = state_t<double>::num_parameters;

  input_t u(0.1, 0.01);

  Dynamics dyn{};

  auto ode = [&](const state_t<double> & x, dstate_t<double> & vel, const double) {
      std::array<double, kStateSize> map_mem;
      static Eigen::Map<state_t<double>> map(map_mem.data());
      static Eigen::Map<const state_t<double>> cmap(map_mem.data());
      map = x;
      vel = dyn(cmap, u);
    };

  cbr::PoseGraphEstimatorParams params;
  params.history = std::chrono::milliseconds(100);
  params.buffer_size = 10;
  params.integration_step = 0.1;
  params.linearize_dynamics = false;

  // no noise
  {
    DubinsPGE filter(std::make_shared<Dynamics>(), params);
    state_t<double> x{};

    boost::numeric::odeint::integrate_const(
      cbr::lie::odeint::euler<state_t<double>, double, dstate_t<double>>(),
      ode, x, 0., 0.5, 0.01,
      [&filter, &u, &gauss](const state_t<double> & x, double t)
      {
        auto t_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(t)
        );

        filter.add_measurement(
          t_ns, std::make_unique<PosRes>(std::get<0>(x).translation(), Sophus::Vector2d{0.1, 0.1})
        );
        auto[xhat, sigma] = filter(t_ns, u);

        ASSERT_LE((std::get<0>(x).inverse() * std::get<0>(xhat)).log().norm(), 1e-3);
      }
    );
  }

  // with noise
  {
    DubinsPGE filter(std::make_shared<Dynamics>(), params);
    state_t<double> x{};

    boost::numeric::odeint::integrate_const(
      cbr::lie::odeint::euler<state_t<double>, double, dstate_t<double>>(),
      ode, x, 0., 0.5, 0.01,
      [&filter, &u, &gauss](const state_t<double> & x, double t)
      {
        auto t_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(t)
        );

        Sophus::Vector2d u_noise = u + 0.1 * Sophus::Vector2d::NullaryExpr(gauss);

        Sophus::Vector2d meas = std::get<0>(x).translation() +
        0.1 * Sophus::Vector2d::NullaryExpr(gauss);

        filter.add_measurement(t_ns, std::make_unique<PosRes>(meas, Sophus::Vector2d{0.1, 0.1}));
        auto[xhat, sigma] = filter(t_ns, u_noise);

        ASSERT_LE((std::get<0>(x).inverse() * std::get<0>(xhat)).log().norm(), 1e-1);
      }
    );
  }
}


TEST(PGE, DubinsExampleLinearize)
{
  spdlog::set_level(spdlog::level::off);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution;
  auto gauss = [&](int) {return distribution(generator);};

  constexpr int kStateSize = state_t<double>::num_parameters;

  input_t u(0.1, 0.01);

  Dynamics dyn{};

  auto ode = [&](const state_t<double> & x, dstate_t<double> & vel, const double) {
      std::array<double, kStateSize> map_mem;
      static Eigen::Map<state_t<double>> map(map_mem.data());
      static Eigen::Map<const state_t<double>> cmap(map_mem.data());
      map = x;
      vel = dyn(cmap, u);
    };

  cbr::PoseGraphEstimatorParams params;
  params.buffer_size = 10;
  params.history = std::chrono::milliseconds(100);
  params.integration_step = 0.1;
  params.linearize_dynamics = true;

  // no noise
  {
    DubinsPGE filter(std::make_shared<Dynamics>(), params);
    state_t<double> x{};

    boost::numeric::odeint::integrate_const(
      cbr::lie::odeint::euler<state_t<double>, double, dstate_t<double>>(),
      ode, x, 0., 0.5, 0.01,
      [&filter, &u, &gauss](const state_t<double> & x, double t)
      {
        auto t_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(t)
        );

        filter.add_measurement(
          t_ns, std::make_unique<PosRes>(std::get<0>(x).translation(), Sophus::Vector2d{0.1, 0.1})
        );
        auto[xhat, sigma] = filter(t_ns, u);

        ASSERT_LE((std::get<0>(x).inverse() * std::get<0>(xhat)).log().norm(), 1e-3);
      }
    );
  }

  // with noise
  {
    DubinsPGE filter(std::make_shared<Dynamics>(), params);
    state_t<double> x{};

    boost::numeric::odeint::integrate_const(
      cbr::lie::odeint::euler<state_t<double>, double, dstate_t<double>>(),
      ode, x, 0., 0.5, 0.01,
      [&filter, &u, &gauss](const state_t<double> & x, double t)
      {
        auto t_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(t)
        );

        Sophus::Vector2d u_noise = u + 0.1 * Sophus::Vector2d::NullaryExpr(gauss);

        Sophus::Vector2d meas = std::get<0>(x).translation() +
        0.1 * Sophus::Vector2d::NullaryExpr(gauss);

        filter.add_measurement(t_ns, std::make_unique<PosRes>(meas, Sophus::Vector2d{0.1, 0.1}));
        auto[xhat, sigma] = filter(t_ns, u_noise);

        ASSERT_LE((std::get<0>(x).inverse() * std::get<0>(xhat)).log().norm(), 1e-1);
      }
    );
  }
}

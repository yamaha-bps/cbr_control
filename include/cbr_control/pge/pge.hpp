// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__PGE__PGE_HPP_
#define CBR_CONTROL__PGE__PGE_HPP_

#include <boost/circular_buffer.hpp>
#include <boost/numeric/odeint.hpp>

#include <ceres/ceres.h>

#include <spdlog/spdlog.h>

#include <cbr_math/lie/odeint.hpp>
#include <cbr_math/interp/piecewise_constant.hpp>

#include <variant>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "ceres_utils.hpp"
#include "ceres_marginalization.hpp"

namespace cbr
{

/**
 * Given a residual BaseResidual(x), create a residual SplitResidual(x1, x2) where x1 and x2
 * are the two nodes closest to x.
 *
 * The cost is s.t. SplitResidual(x1, x2) = BaseResidual( interp(alpha, x1, x2) )
 * for an interpolation of x1, x2 where t1 <= t < t2
 *
 * @tparam BaseResidual single-state residual function
 * @param alpha interpolation value in interval [0, 1)
 * @param base instance of BaseResidual to utilize (this class takes ownership)
 */
template<template<typename> typename State, typename BaseResidual>
class SplitResidual
{
public:
  explicit SplitResidual(double alpha, std::unique_ptr<BaseResidual> base)
  : alpha_(alpha),
    base_(std::move(base))
  {}

  template<typename T>
  auto operator()(Eigen::Map<const State<T>> x1, Eigen::Map<const State<T>> x2) const
  {
    // need to pass a Map<const State> to base
    std::array<T, State<T>::num_parameters> tmp;
    Eigen::Map<State<T>> x_tmp(tmp.data());
    Eigen::Map<const State<T>> cx_tmp(tmp.data());
    // interpolate with constant velocity, consistent with dynamics if we use Euler integration...
    x_tmp = x1 * State<T>::exp(alpha_ * (x1.inverse() * x2).log());
    return base_->template operator()<T>(cx_tmp);
  }

private:
  double alpha_;
  std::unique_ptr<BaseResidual> base_;
};

/**
 * Dynamics residual
 */
template<template<typename> typename State_, typename Input_, typename Dynamics_>
class DynamicsResidual
{
  static constexpr int nx = State_<double>::DoF;
  static constexpr int nu = Input_::SizeAtCompileTime;

  using u_queue_t = boost::circular_buffer<
    std::pair<std::chrono::nanoseconds, Input_>,
    Eigen::aligned_allocator<std::pair<std::chrono::nanoseconds, Input_>>
  >;

public:
  /**
   * @brief Construct a new DynamicsResidual
   *
   * @param dynamics pointer to dynamics containing ode<T> function
   * @param u_queue sorted inputs
   * @param t0 start of interval
   * @param tf end of interval
   * @param dt integration timestep
   *
   * If u_queue is empty zero input is assumed
   * Must connect the entire state
   */
  DynamicsResidual(
    std::shared_ptr<Dynamics_> dynamics,
    const u_queue_t & u_queue,
    const std::chrono::nanoseconds & t0,
    const std::chrono::nanoseconds & tf,
    const double dt,
    const std::optional<State_<double>> & lin_state_ = {}
  )
  : dyn_(dynamics),
    tspan_(std::chrono::duration_cast<std::chrono::duration<double>>(tf - t0).count()),
    dt_(std::min<double>(dt, tspan_)),
    input_fit_{std::nullopt}
  {
    if (u_queue.empty()) {
      // assume input is zero if none is given
      input_fit_ = cbr::PiecewiseConstant::fitND(
        Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(1),
        Eigen::MatrixXd::Zero(static_cast<int>(Input_::SizeAtCompileTime), 1)
      );
    } else {
      // fit spline to inputs
      Eigen::Matrix<double, 1, Eigen::Dynamic> x(u_queue.size());
      Eigen::MatrixXd y(static_cast<int>(Input_::SizeAtCompileTime), u_queue.size());

      std::size_t i = 0;
      for (const auto &[t, u] : u_queue) {
        x(i) = std::chrono::duration_cast<std::chrono::duration<double>>(t - t0).count();
        y.col(i) = u;
        ++i;
      }

      input_fit_ = cbr::PiecewiseConstant::fitND(std::move(x), y);
    }

    // linearize dynamics around state
    // approximately holds that f(x) = A * (xl.inv() * x).log() + B * (u - ul) + K
    if (lin_state_.has_value()) {
      using Tad = autodiff::forward::dual;

      State_<Tad> x_ad = lin_state_.value().template cast<Tad>();
      auto u_ad = input_fit_.value().template val<Tad>(Tad(0));

      Eigen::Matrix<Tad, nx, 1> dx_ad;
      dx_ad.setZero();


      auto fcn = [&](const auto & var_dx, const auto & var_u) -> Eigen::Matrix<Tad, nx, 1> {
          return dyn_->operator()(x_ad * State_<Tad>::exp(var_dx), var_u);
        };

      Eigen::Matrix<Tad, nx, 1> f_ad;

      Eigen::Matrix<double, nx, nx> A = autodiff::forward::jacobian(
        fcn, autodiff::forward::wrt(dx_ad), autodiff::forward::at(dx_ad, u_ad), f_ad
      );
      Eigen::Matrix<double, nx, nu> B = autodiff::forward::jacobian(
        fcn, autodiff::forward::wrt(u_ad), autodiff::forward::at(dx_ad, u_ad), f_ad
      );
      Eigen::Matrix<double, nx, 1> K = f_ad.template cast<double>();

      lin_point_ = lin_state_;
      lin_dyn_.emplace(std::make_tuple(A, B, K));
    }
  }

  template<typename T>
  typename State_<T>::Tangent
  operator()(const Eigen::Map<const State_<T>> x0, const Eigen::Map<const State_<T>> x1)
  {
    State_<T> x_pred = integrate<T>(x0);
    Eigen::Matrix<double, nx, nx> sqrt_inf = dyn_->sqrt_inf(x_pred.template cast<double>());
    // Approximation that noise is dominated by most recent dynamics updates
    sqrt_inf /= tspan_;
    return sqrt_inf.template cast<T>() * (x_pred.inverse() * x1).log();
  }

  template<typename T>
  State_<T> integrate(const Eigen::Map<const State_<T>> x0)
  {
    State_<T> x_pred(x0);
    boost::numeric::odeint::integrate_const(
      lie::odeint::euler<State_<T>, double, typename State_<T>::Tangent>{},
      std::bind(
        &DynamicsResidual::template system<T>, this, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3
      ),
      x_pred, 0., tspan_, dt_
    );
    return x_pred;
  }

private:
  // evaluate right-hand side of ode: deriv = f(x, u(t))
  template<typename T>
  void system(const State_<T> & state, typename State_<T>::Tangent & deriv, const double t) const
  {
    if (lin_dyn_.has_value() && lin_point_.has_value()) {
      const auto & [A, B, K] = lin_dyn_.value();

      const Eigen::Matrix<T, nx,
        1> dx = (lin_point_.value().inverse().template cast<T>() * state).log();
      const Eigen::Matrix<T, nu, 1> du =
        input_fit_.value().template val<T>(t) - input_fit_.value().template val<double>(0.);

      deriv = A * dx + B * du + K;
    } else {
      deriv = dyn_->operator()(state, input_fit_.value().template val<double>(t));
    }
  }

  // member variables
  std::shared_ptr<Dynamics_> dyn_;

  std::optional<State_<double>> lin_point_;
  std::optional<std::tuple<
      Eigen::Matrix<double, nx, nx>,
      Eigen::Matrix<double, nx, nu>,
      Eigen::Matrix<double, nx, 1>
    >> lin_dyn_{};

  double tspan_;
  double dt_;
  std::optional<cbr::PiecewisePolyND> input_fit_;
};


/**
 * Marginalization factor attached to a single state x0
 * @param mu nominal value for x0
 * @param gamma value of x0 \ominus mu at linearization point
 * @param sqrt_inf square root information matrix
 */
template<template<typename> typename State>
class MarginalizationResidual
{
public:
  static constexpr int DoF = State<double>::DoF;

  MarginalizationResidual(
    const State<double> & mu,
    Eigen::Matrix<double, DoF, 1> && gamma, Eigen::Matrix<double, DoF, DoF> && sqrt_inf)
  : mu_(mu), gamma_(std::move(gamma)), sqrt_inf_(std::move(sqrt_inf))
  {}

  template<typename T>
  bool operator()(T const * const * params, T * residuals)
  {
    Eigen::Map<const State<T>> x(params[0]);
    Eigen::Map<Eigen::Matrix<T, DoF, 1>> res(residuals);
    res = sqrt_inf_ * ((x * mu_.inverse()).log() - gamma_);
    return true;
  }

private:
  const State<double> mu_;
  const Eigen::Matrix<double, DoF, 1> gamma_;
  const Eigen::Matrix<double, DoF, DoF> sqrt_inf_;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


struct PoseGraphEstimatorParams
{
  std::chrono::nanoseconds history{100000};   // max interval of history to keep
  std::size_t buffer_size{10};     // max number of states to keep
  double integration_step{0.1};    // size of timestep for dynamics integration

  bool linearize_dynamics{false};  // linearize dynamics on residual construction
  bool log{false};  // print logging information
};


/**
 * Pose graph inspired estimator for dynamical systems
 * @tparam State_ Lie group type representing system state templated on scalar type
 * @tparam Input_ Type system input: must be Eigen vector type
 * @tparam Dynamics_ class with function "State ode()(const State_ &, const Input_ &)" representing system dynamics
 * @tparam MeasurementResiduals_ ... classes with "operator()(const State &)" representing state measurements
 */
template<
  template<typename> typename State_,
  typename Input_,
  typename Dynamics_,
  typename ... MeasurementResiduals_
>
class PoseGraphEstimator
{
public:
  template<typename Scalar>
  using State = State_<Scalar>;
  static constexpr int DoF = State_<double>::DoF;
  static constexpr int num_parameters = State_<double>::num_parameters;
  using Input = Input_;
  using x_t = std::array<double, State_<double>::num_parameters>;
  using tx_t = std::pair<std::chrono::nanoseconds, x_t>;
  using buffer_t = boost::circular_buffer<tx_t>;
  using meas_t = std::variant<std::unique_ptr<MeasurementResiduals_>...>;
  using cov_t = Eigen::Matrix<double, DoF, DoF>;

  PoseGraphEstimator() = delete;
  PoseGraphEstimator(const PoseGraphEstimator &) = delete;
  PoseGraphEstimator(PoseGraphEstimator &&) = delete;
  PoseGraphEstimator & operator=(const PoseGraphEstimator &) = delete;
  PoseGraphEstimator & operator=(PoseGraphEstimator &&) = delete;

  /**
   * PoseGraphEstimator: dynamical state estimator using arbitraty measurement functions
   * @param dynamics
   * @param params
   * @param initial_state optional initial state
   * The oldest state is marginalized when they become older than history, or when the buffer is filled,
   * whichever happens first.
   */
  PoseGraphEstimator(
    std::shared_ptr<Dynamics_> dynamics,
    PoseGraphEstimatorParams params,
    State<double> initial_state = State<double>{}
  )
  : dynamics_(std::move(dynamics)),
    prm_(params),
    buffer_(prm_.buffer_size),
    initial_state_(initial_state),
    meas_queue_{},
    rel_meas_queue_{},
    input_queue_(100),
    parameterization_(std::make_unique<LieGroupParameterization<State<double>>>()
    )
  {
    ceres::Problem::Options opts;
    opts.enable_fast_removal = true;

    // let ceres handle deletion of cost and loss functions
    opts.cost_function_ownership = ceres::Ownership::TAKE_OWNERSHIP;
    opts.loss_function_ownership = ceres::Ownership::TAKE_OWNERSHIP;

    // we handle the parameterization since there's a single one
    opts.local_parameterization_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;

    opts.disable_all_safety_checks = true;
    problem_ = std::make_unique<ceres::Problem>(opts);

    if (prm_.log) {
      std::cout << "Created PGE" << std::endl;
    }
  }

  ~PoseGraphEstimator() = default;

  /**
   * Update filter and return most recent estimate and the covariance in tangent space
   * @param t time to estimate state at
   * @param u_last (constant) input applied since last call to this function
   */
  std::pair<State<double>, cov_t> operator()(std::chrono::nanoseconds t, const Input & u_last)
  {
    if (!buffer_.empty()) {
      this->add_input(buffer_.back().first, u_last);
    } else {
      this->add_input(t, u_last);
    }
    return this->operator()(t);
  }

  /**
   * Update filter and return most recent estimate and the covariance in tangent space
   * @param t time to estimate state at
   */
  std::pair<State<double>, cov_t> operator()(std::chrono::nanoseconds t)
  {
    std::lock_guard lock(mtx_);

    if (buffer_.empty()) {
      // first iteration
      buffer_.push_back({t, x_t{}});
      Eigen::Map<State<double>> map(buffer_.back().second.data());

      problem_->AddParameterBlock(
        buffer_.back().second.data(), State<double>::num_parameters, parameterization_.get()
      );

      // initial guess
      map = initial_state_;

      // add initial marginalization factor
      Eigen::Matrix<double, DoF, 1> gamma_init = Eigen::Matrix<double, DoF, 1>::Zero();
      Eigen::Matrix<double, DoF, 1> stdev_init = Eigen::Matrix<double, DoF, 1>::Ones();
      auto mcost = new ceres::DynamicAutoDiffCostFunction<MarginalizationResidual<State>>(
        new MarginalizationResidual<State>(
          initial_state_, std::move(gamma_init), stdev_init.cwiseInverse().asDiagonal()
      ));
      mcost->AddParameterBlock(num_parameters);
      mcost->SetNumResiduals(DoF);
      problem_->AddResidualBlock(mcost, NULL, {buffer_.back().second.data()});

    } else {
      if (t <= buffer_.back().first) {
        std::cerr << "pge() must be called with strictly increasing times";
        return {
          State<double>(Eigen::Map<State<double>>(buffer_.back().second.data())), cov_t::Identity()
        };
      }

      buffer_.push_back({t, x_t{}});
      problem_->AddParameterBlock(
        buffer_.back().second.data(), State<double>::num_parameters, parameterization_.get()
      );

      size_t N = buffer_.size();

      std::chrono::nanoseconds t0 = buffer_[N - 2].first;
      std::chrono::nanoseconds tf = buffer_[N - 1].first;

      double * node_km1 = buffer_[N - 2].second.data();
      double * node_k = buffer_[N - 1].second.data();

      Eigen::Map<const State<double>> map_km1(node_km1);
      Eigen::Map<State<double>> map_k(node_k);

      // add dynamics residual from node_km1 to node_k
      {
        std::lock_guard lock(input_mtx_);

        // remove old inputs to keep at most one before interval [t_km1, t_k]
        while (input_queue_.size() > 1 && input_queue_[1].first < t0) {
          input_queue_.pop_front();
        }

        // Add dynamics residual with queue of inputs received since last estimate
        std::optional<State_<double>> lin_state{};
        if (prm_.linearize_dynamics) {
          lin_state = map_km1;  // linearize around last point
        }
        auto dynamics_residual = std::make_unique<DynamicsResidual<State_, Input_, Dynamics_>>(
          dynamics_, input_queue_, t0, tf, prm_.integration_step, lin_state
        );

        // set initial guess from dynamics integration
        map_k = dynamics_residual->template integrate<double>(map_km1);

        auto dyn_cost_fcn =
          AutodiffCostWrapper<DynamicsResidual<State_, Input_, Dynamics_>
          >::create(std::move(dynamics_residual));
        problem_->AddResidualBlock(dyn_cost_fcn, NULL, node_km1, node_k);
      }


      // add queued absolute measurements
      {
        std::lock_guard lock(meas_mtx_);
        for (auto & item : meas_queue_) {
          std::visit(
            [this, &item](
              auto && residual) {
              using residual_t = typename std::remove_reference_t<decltype(residual)>::element_type;
              using argument_t =
              typename signature<decltype(&residual_t::template operator()<double>)>::argument_type;
              if constexpr (argument_t::size == 1) {
                add_measurement_(item.first, std::move(residual));
              }
            }, item.second);
        }
        meas_queue_.clear();
      }

      // add queued relative measurements
      {
        std::lock_guard lock(rel_meas_mtx_);
        for (auto & item : rel_meas_queue_) {
          std::visit(
            [this, &item](
              auto && residual) {
              using residual_t = typename std::remove_reference_t<decltype(residual)>::element_type;
              using argument_t =
              typename signature<decltype(&residual_t::template operator()<double>)>::argument_type;
              if constexpr (argument_t::size == 2) {
                add_relative_measurement_(
                  item.first.first, item.first.second,
                  std::move(residual));
              }
            }, item.second);
        }
        rel_meas_queue_.clear();
      }
    }

    // optimize
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.dense_linear_algebra_library_type = ceres::LAPACK;
    options.max_num_iterations = 20;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = prm_.log;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem_.get(), &summary);
    if (prm_.log) {
      std::cout << summary.FullReport();
    }

    // If solution isn't usable, return 0 matrix for covariance
    if (!summary.IsSolutionUsable()) {
      return {State<double>{}, cov_t::Zero()};
    }

    // Estimate covariance
    ceres::Covariance::Options covariance_options;
    covariance_options.algorithm_type = ceres::DENSE_SVD;
    covariance_options.null_space_rank = -1;
    ceres::Covariance covariance(covariance_options);
    std::vector<std::pair<const double *, const double *>> covariance_blocks{
      {buffer_.back().second.data(), buffer_.back().second.data()}
    };
    std::array<double, DoF * DoF> covariance_kk;
    covariance.Compute(covariance_blocks, problem_.get());
    covariance.GetCovarianceBlockInTangentSpace(
      buffer_.back().second.data(),
      buffer_.back().second.data(), covariance_kk.data());
    cov_t cov_mat(covariance_kk.data());

    if (prm_.log) {
      std::cout << summary.BriefReport() << std::endl;
    }


    // marginalize old nodes
    std::chrono::nanoseconds cutoff = buffer_.back().first - prm_.history;
    while (buffer_.full() || buffer_.front().first < cutoff) {
      double * node = buffer_.front().second.data();

      std::vector<MarginalizationInfo> varinfo;
      Eigen::VectorXd gamma_dyn;
      Eigen::MatrixXd sqrt_inf_dyn;
      bool succ = marginalize(problem_.get(), node, varinfo, gamma_dyn, sqrt_inf_dyn);

      if (!succ || gamma_dyn.rows() != DoF || sqrt_inf_dyn.cols() != DoF ||
        varinfo.size() != 1 || varinfo[0].gsize != num_parameters || varinfo[0].size != DoF)
      {
        std::cerr << "Unexpected marginalization result, skipping" << std::endl;
      } else {
        Eigen::Map<const State<double>> mu(varinfo[0].mu.data());
        Eigen::Matrix<double, DoF, 1> gamma = gamma_dyn;
        Eigen::Matrix<double, -1, DoF> sqrt_inf = sqrt_inf_dyn;

        auto mcost = new ceres::DynamicAutoDiffCostFunction<MarginalizationResidual<State>>(
          new MarginalizationResidual<State>(
            mu.template cast<double>(), std::move(gamma), std::move(sqrt_inf)
        ));
        mcost->AddParameterBlock(num_parameters);
        mcost->SetNumResiduals(DoF);
        problem_->AddResidualBlock(mcost, NULL, {const_cast<double *>(varinfo[0].p)});
      }

      problem_->RemoveParameterBlock(node);
      buffer_.pop_front();
    }

    return {State<double>(Eigen::Map<State<double>>(buffer_.back().second.data())), cov_mat};
  }

  /**
   * Add an input
   * @param t input timestamp
   * @param input
   */
  void add_input(std::chrono::nanoseconds t, const Input & input)
  {
    std::lock_guard lock(input_mtx_);
    if (!input_queue_.empty() && input_queue_.back().first >= t) {
      std::cerr << "Inputs must be strictly increasing, skipping..." << std::endl;
      return;
    }

    if (input_queue_.full()) {
      if (prm_.log) {
        std::cout << "Input queue full, doubling its size" << std::endl;
      }
      input_queue_.resize(2 * input_queue_.size());
    }
    input_queue_.push_back(std::make_pair(t, input));
  }

  /**
   * Add a measurement
   * @param t time when measurement was taken
   * @param meas measurement conforming to measurement interface
   */
  template<typename Residual>
  void add_measurement(std::chrono::nanoseconds t, std::unique_ptr<Residual> residual)
  {
    // we can directly add it if within bounds and optimization is not running
    if (mtx_.try_lock()) {
      if (t < buffer_.back().first) {
        add_measurement_(t, std::move(residual));
        mtx_.unlock();
        return;
      }
      mtx_.unlock();
    }

    // queue it for addition later
    std::lock_guard lock(meas_mtx_);
    meas_queue_.emplace_back(t, std::move(residual));
  }

  template<typename Residual>
  void add_relative_measurement(
    std::chrono::nanoseconds t1, std::chrono::nanoseconds t2,
    std::unique_ptr<Residual> residual)
  {
    // we can directly add it if within bounds and optimization is not running
    if (mtx_.try_lock()) {
      if (t2 < buffer_.back().first) {
        add_relative_measurement_(t1, t2, std::move(residual));
        mtx_.unlock();
        return;
      }
      mtx_.unlock();
    }

    // queue it for addition later
    std::lock_guard lock(rel_meas_mtx_);
    rel_meas_queue_.emplace_back(std::make_pair(t1, t2), std::move(residual));
  }

protected:
  template<typename Residual>
  void add_relative_measurement_(
    std::chrono::nanoseconds t1, std::chrono::nanoseconds t2,
    std::unique_ptr<Residual> residual)
  {
    if (buffer_.empty()) {
      if (prm_.log) {
        std::cerr << "Buffer empty" << std::endl;
      }
      return;
    }

    const std::chrono::nanoseconds & tmin = buffer_.front().first;
    const std::chrono::nanoseconds & tmax = buffer_.back().first;

    if (tmin <= t1 && t2 <= tmax) {   // add SplitResidual between two states
      // find nodes after start and finish
      auto it1 = buffer_.end() - 1;
      auto it2 = buffer_.begin();
      while (t1 < it1->first) {
        --it1;
      }
      while (t2 > it2->first) {
        ++it2;
      }
      if (it1 != it2) {
        problem_->AddResidualBlock(
          AutodiffCostWrapper<Residual>::create(std::move(residual)),
          NULL, it1->second.data(), it2->second.data()
        );
      } else if (prm_.log) {
        std::cerr <<
          "Measurement w/ t1=" << t1.count() << " t2=" << t2.count() <<
          " attached to same node, dropping" << std::endl;
      }
    } else if (prm_.log) {
      std::cerr <<
        "Measurement w/ t1=" << t1.count() << " t2=" << t2.count() << " not in [" <<
        tmin.count() << "," << tmax.count() << "], dropping" << std::endl;
    }
  }

  /**
   * Internal method for adding measurements to problem_
   * Not thread-safe
   */
  template<typename Residual>
  void add_measurement_(std::chrono::nanoseconds t, std::unique_ptr<Residual> residual)
  {
    if (buffer_.empty()) {
      if (prm_.log) {
        std::cerr << "Buffer empty" << std::endl;
      }
      return;
    }

    const std::chrono::nanoseconds & tmin = buffer_.front().first;
    const std::chrono::nanoseconds & tmax = buffer_.back().first;

    if (tmin <= t && t < tmax) {   // add SplitResidual between two states
      // find node preceding measurement
      auto it = buffer_.end() - 1;
      while (t < it->first) {
        --it;
      }
      std::chrono::nanoseconds dt = (it + 1)->first - it->first;

      double alph =
        std::chrono::duration_cast<std::chrono::duration<double>>(t - it->first).count() /
        std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();

      auto cost_fcn = AutodiffCostWrapper<SplitResidual<State, Residual>>
        ::create(std::make_unique<SplitResidual<State, Residual>>(alph, std::move(residual)));

      problem_->AddResidualBlock(cost_fcn, NULL, it->second.data(), (it + 1)->second.data());
    } else if (t == tmax) {  // add Residual to most recent node
      auto cost_fcn = AutodiffCostWrapper<Residual>::create(std::move(residual));
      problem_->AddResidualBlock(cost_fcn, NULL, buffer_.back().second.data());
    } else if (prm_.log) {
      std::cerr << "Measurement w/ t=" << t.count() << " not in [" << tmin.count() << "," <<
        tmax.count() << "], dropping" << std::endl;
    }
  }

private:
  std::shared_ptr<Dynamics_> dynamics_;

  PoseGraphEstimatorParams prm_;

  buffer_t buffer_;

  State<double> initial_state_;

  std::mutex mtx_;  // locked by update loop
  std::unique_ptr<ceres::Problem> problem_;

  std::mutex meas_mtx_;  // protects meas_queue_
  std::vector<std::pair<std::chrono::nanoseconds, meas_t>> meas_queue_;

  std::mutex rel_meas_mtx_;  // protects meas_queue_
  std::vector<std::pair<std::pair<std::chrono::nanoseconds, std::chrono::nanoseconds>,
    meas_t>> rel_meas_queue_;

  std::mutex input_mtx_;
  boost::circular_buffer<
    std::pair<std::chrono::nanoseconds, Input>,
    Eigen::aligned_allocator<std::pair<std::chrono::nanoseconds, Input>>
  > input_queue_;

  const std::unique_ptr<ceres::LocalParameterization> parameterization_;
};

}  // namespace cbr

#endif  // CBR_CONTROL__PGE__PGE_HPP_

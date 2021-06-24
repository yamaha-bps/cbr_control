// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__PID_HPP_
#define CBR_CONTROL__PID_HPP_

#include <boost/fusion/adapted/struct.hpp>

#include <cbr_utils/cyber_timer.hpp>

#include <sophus/so2.hpp>

#include <memory>
#include <type_traits>
#include <utility>

#include "cbr_control/derivator.hpp"

namespace cbr
{

struct PIDClampParams
{
  bool active{true};
  double max{1.};
  double min{-1.};

  void check_correctness() const
  {
    if (min > max) {
      throw std::invalid_argument("max must be less or equal to min");
    }
  }
};

struct PIDParams
{
  double P{1.};
  double I{1.};
  double D{1.};

  double der_tau{0.};

  PIDClampParams clamp;

  void check_correctness() const
  {
    clamp.check_correctness();
  }
};

template<typename _clock_t = std::chrono::high_resolution_clock>
class PID
{
public:
  using clock_t = _clock_t;
  using timer_t = CyberTimerNoAvg<std::ratio<1>, double, clock_t>;
  using derivator_t = Derivator<double, clock_t>;

  // Constructors
  PID() = default;
  PID(const PID &) = default;
  PID(PID &) = default;
  PID(PID &&) = default;
  PID & operator=(const PID &) = default;
  PID & operator=(PID &) = default;
  PID & operator=(PID &&) = default;
  ~PID() = default;

  template<typename T>
  explicit PID(T && clock, const PIDParams & prm = {}, const double errInt = 0.)
  : timer_(std::forward<T>(clock)),
    prm_(prm),
    errInt_(errInt)
  {
    prm_.check_correctness();
    der_.set_params({prm_.der_tau});
  }

  explicit PID(const PIDParams & prm, const double errInt = 0.)
  : prm_(prm),
    errInt_(errInt)
  {
    prm_.check_correctness();
    der_.set_params({prm_.der_tau});
  }

  template<typename T>
  PID(T && clock, const double errInt)
  : timer_(std::forward<T>(clock)),
    errInt_(errInt)
  {}

  explicit PID(const double errInt)
  : errInt_(errInt)
  {}

  template<typename T>
  void set_clock(T && clock)
  {
    timer_.set_clock(std::forward<T>(clock));
  }

  void set_params(const PIDParams & prm)
  {
    prm.check_correctness();
    prm_ = prm;
    der_.set_params({prm_.der_tau});
  }

  const PIDParams & get_params() const
  {
    return prm_;
  }

  const double & update(const double desired, const double actual)
  {
    return update_impl(desired - actual, actual);
  }

  const double & update(const Sophus::SO2d & desired, const Sophus::SO2d & actual)
  {
    return update_impl((actual.inverse() * desired).log(), actual.log());
  }

  const double & operator()(const double desired, const double actual)
  {
    return update(desired, actual);
  }

  const double & operator()(const Sophus::SO2d & desired, const Sophus::SO2d & actual)
  {
    return update(desired, actual);
  }

  const double & getValue() const
  {
    return output_;
  }

  void reset(const double errInt = 0.)
  {
    der_.reset();
    errInt_ = prm_.I * errInt;
    if (prm_.clamp.active) {
      errInt_ = std::clamp(errInt_, prm_.clamp.min, prm_.clamp.max);
    }
    output_ = errInt_;
    timer_.tic();
  }

protected:
  const double & update_impl(const double err, const double actual)
  {
    const auto tNow = timer_.now();
    const double errDot = der_.update(actual, tNow);
    output_ = prm_.P * err + prm_.D * errDot + errInt_;

    if (init_) {
      const double dt = timer_.toctic(tNow);
      const double dt_err = dt * err * prm_.I;
      const double errIntTmp = errInt_ + dt_err;
      const double outputTmp = output_ + dt_err;
      if (!prm_.clamp.active || (outputTmp < prm_.clamp.max && outputTmp > prm_.clamp.min)) {
        errInt_ = errIntTmp;
        output_ = outputTmp;
      }
    } else {
      init_ = true;
      timer_.tic(tNow);
    }

    if (prm_.clamp.active) {
      output_ = std::clamp(output_, prm_.clamp.min, prm_.clamp.max);
    }
    return output_;
  }

  timer_t timer_ = timer_t(clock_t{});
  PIDParams prm_;
  double errInt_{0.};
  double output_{0.};
  bool init_{false};
  derivator_t der_;
};

}    // namespace cbr

// cppcheck-suppress unknownMacro
BOOST_FUSION_ADAPT_STRUCT(
  cbr::PIDClampParams,
  active,
  max,
  min
)

// cppcheck-suppress unknownMacro
BOOST_FUSION_ADAPT_STRUCT(
  cbr::PIDParams,
  P,
  I,
  D,
  der_tau,
  clamp
)

#endif  // CBR_CONTROL__PID_HPP_

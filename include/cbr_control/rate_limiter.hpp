// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__RATE_LIMITER_HPP_
#define CBR_CONTROL__RATE_LIMITER_HPP_

#include <boost/hana/adapt_struct.hpp>

#include <cbr_utils/cyber_timer.hpp>

#include <memory>
#include <type_traits>
#include <utility>


namespace cbr
{

struct RateLimiterParams
{
  double rise_rate{1.};
  double fall_rate{-1.};

  void check_correctness() const
  {
    if (rise_rate < 0.) {
      throw std::invalid_argument("parameter 'rise_rate' must be >= 0");
    }
    if (fall_rate > 0.) {
      throw std::invalid_argument("parameter 'fall_rate' must be <= 0");
    }
  }
};

template<typename _clock_t = std::chrono::high_resolution_clock>
class RateLimiter
{
public:
  using clock_t = _clock_t;
  using timer_t = CyberTimerNoAvg<std::ratio<1>, double, clock_t>;

  // Constructors
  RateLimiter() = default;
  RateLimiter(const RateLimiter &) = default;
  RateLimiter(RateLimiter &) = default;
  RateLimiter(RateLimiter &&) = default;
  RateLimiter & operator=(const RateLimiter &) = default;
  RateLimiter & operator=(RateLimiter &) = default;
  RateLimiter & operator=(RateLimiter &&) = default;
  ~RateLimiter() = default;

  template<typename T>
  explicit RateLimiter(T && clock, const RateLimiterParams & prm = {})
  : timer_(std::forward<T>(clock)),
    prm_(prm)
  {
    prm_.check_correctness();
  }

  explicit RateLimiter(const RateLimiterParams & prm)
  : prm_(prm)
  {
    prm_.check_correctness();
  }

  template<typename T>
  void set_clock(T && clock)
  {
    timer_.set_clock(std::forward<T>(clock));
  }

  void set_params(const RateLimiterParams & prm)
  {
    prm.check_correctness();
    prm_ = prm;
  }

  const RateLimiterParams & get_params() const
  {
    return prm_;
  }

  const double & update(const double val, const typename timer_t::time_point tNow)
  {
    if (init_) {
      // Define dt
      const double dt = timer_.toctic(tNow);

      // Compute Rate (derivative)
      if (dt <= 0.) {
        return output_;
      }
      const double rate = (val - output_) / dt;

      // Assign Output
      if (rate > prm_.rise_rate) {
        return output_ += dt * prm_.rise_rate;
      } else if (rate < prm_.fall_rate) {
        return output_ += dt * prm_.fall_rate;
      } else {
        return output_ = val;
      }
    }

    init_ = true;
    timer_.tic(tNow);

    return output_ = val;
  }

  const double & update(const double val)
  {
    return update(val, timer_.now());
  }

  const double & operator()(const double val)
  {
    return update(val);
  }

  const double & getValue() const
  {
    return output_;
  }

  void reset()
  {
    init_ = false;
  }

  const double & reset(const double val, const typename timer_t::time_point tNow)
  {
    init_ = true;
    timer_.tic(tNow);
    return output_ = val;
  }

  const double & reset(const double val)
  {
    return reset(val, timer_.now());
  }

protected:
  timer_t timer_ = timer_t(clock_t{});
  RateLimiterParams prm_;
  double output_{0.};
  bool init_{false};
};

}    // namespace cbr

// cppcheck-suppress unknownMacro
BOOST_HANA_ADAPT_STRUCT(
  cbr::RateLimiterParams,
  rise_rate,
  fall_rate
);

#endif  // CBR_CONTROL__RATE_LIMITER_HPP_

// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__INTEGRATOR_HPP_
#define CBR_CONTROL__INTEGRATOR_HPP_

#include <boost/hana/adapt_struct.hpp>

#include <cbr_utils/cyber_timer.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace cbr
{

struct IntegratorParams
{
  bool saturate{false};
  double sat_min{-1.};
  double sat_max{1.};

  void check_correctness() const
  {
    if (sat_min > sat_max) {
      throw std::invalid_argument("sat_min must be less or equal to sat_max");
    }
  }
};

template<typename _clock_t = std::chrono::high_resolution_clock>
class Integrator
{
public:
  using clock_t = _clock_t;
  using timer_t = CyberTimerNoAvg<std::ratio<1>, double, clock_t>;

  Integrator() = default;
  Integrator(const Integrator &) = default;
  Integrator(Integrator &) = default;
  Integrator(Integrator &&) noexcept = default;
  Integrator & operator=(const Integrator &) = default;
  Integrator & operator=(Integrator &&) noexcept = default;
  ~Integrator() = default;

  template<typename T>
  explicit Integrator(T && clock, const IntegratorParams & prm = {}, const double valNm1 = 0.)
  : timer_(std::forward<T>(clock)),
    prm_(prm),
    valNm1_(valNm1)
  {
    prm_.check_correctness();
  }

  explicit Integrator(const IntegratorParams & prm, const double valNm1 = 0.)
  : prm_(prm),
    valNm1_(valNm1)
  {
    prm_.check_correctness();
  }

  template<typename T>
  Integrator(T && clock, const double valNm1)
  : timer_(std::forward<T>(clock)),
    valNm1_(valNm1)
  {}

  explicit Integrator(const double valNm1)
  : valNm1_(valNm1)
  {}

  template<typename T>
  void set_clock(T && clock)
  {
    timer_.set_clock(std::forward<T>(clock));
  }

  void set_params(const IntegratorParams & prm)
  {
    prm.check_correctness();
    prm_ = prm;
  }

  const IntegratorParams & get_params() const
  {
    return prm_;
  }

  const double & update(const double val, const typename timer_t::time_point tNow)
  {
    if (init_) {
      const double dt = timer_.toctic(tNow);

      valNm1_ += dt * val;

      if (prm_.saturate) {
        valNm1_ = std::clamp(valNm1_, prm_.sat_min, prm_.sat_max);
      }
    }

    init_ = true;
    timer_.tic(tNow);
    return valNm1_;
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
    return valNm1_;
  }

  const double & reset(const double val, const typename timer_t::time_point tNow)
  {
    valNm1_ = val;
    if (prm_.saturate) {
      valNm1_ = std::clamp(valNm1_, prm_.sat_min, prm_.sat_max);
    }
    timer_.tic(tNow);
    init_ = true;
    return valNm1_;
  }

  const double & reset(const typename timer_t::time_point tNow)
  {
    return reset(0., tNow);
  }

  const double & reset(const double val)
  {
    return reset(val, timer_.now());
  }

  const double & reset()
  {
    return reset(0., timer_.now());
  }

protected:
  timer_t timer_ = timer_t(clock_t{});
  IntegratorParams prm_;
  double valNm1_ = 0.;
  bool init_{false};
};

}  // namespace cbr

// cppcheck-suppress unknownMacro
BOOST_HANA_ADAPT_STRUCT(
  cbr::IntegratorParams,
  saturate,
  sat_min,
  sat_max
);

#endif  // CBR_CONTROL__INTEGRATOR_HPP_

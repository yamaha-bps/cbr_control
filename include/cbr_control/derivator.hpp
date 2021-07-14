// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__DERIVATOR_HPP_
#define CBR_CONTROL__DERIVATOR_HPP_

#include <boost/hana/adapt_struct.hpp>

#include <cbr_utils/cyber_timer.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace cbr
{

struct DerivatorParams
{
  double filter_tau{0.};

  void check_correctness() const
  {
  }
};

template<typename T, typename _clock_t = std::chrono::high_resolution_clock>
class Derivator
{
public:
  using clock_t = _clock_t;
  using timer_t = CyberTimerNoAvg<std::ratio<1>, double, clock_t>;

  Derivator() = default;
  Derivator(const Derivator &) = default;
  Derivator(Derivator &) = default;
  Derivator(Derivator &&) noexcept = default;
  Derivator & operator=(const Derivator &) = default;
  Derivator & operator=(Derivator &&) noexcept = default;
  ~Derivator() = default;

  template<typename T1>
  explicit Derivator(T1 && clock, const DerivatorParams & prm = {})
  : timer_(std::forward<T1>(clock)),
    prm_(prm)
  {
    prm_.check_correctness();
  }

  explicit Derivator(const DerivatorParams & prm)
  : prm_(prm)
  {
    prm_.check_correctness();
  }

  template<typename T1>
  void set_clock(T1 && clock)
  {
    timer_.set_clock(std::forward<T1>(clock));
  }

  void set_params(const DerivatorParams & prm)
  {
    prm.check_correctness();
    prm_ = prm;
  }

  const DerivatorParams & get_params() const
  {
    return prm_;
  }

  const T & update(const T & val, const typename timer_t::time_point tNow)
  {
    if (init_) {
      const double dt = timer_.toctic(tNow);

      if (prm_.filter_tau <= 0.) {  // no filtering
        if (dt == 0.) {
          derNm1_ = val - valNm1_;
        } else {
          derNm1_ = (val - valNm1_) / dt;
        }
        valNm1_ = val;
        return derNm1_;
      }
      const double e = std::exp(-dt / prm_.filter_tau);
      derNm1_ = (val - valNm1_) * e / prm_.filter_tau;
      valNm1_ = valNm1_ * e + (1 - e) * val;
      return derNm1_;
    }

    timer_.tic(tNow);
    init_ = true;
    valNm1_ = val;
    derNm1_ = 0.;
    return derNm1_;
  }

  const T & update(const T & val)
  {
    return update(val, timer_.now());
  }

  const T & operator()(const T & val)
  {
    return update(val);
  }

  const T & getValue() const
  {
    return derNm1_;
  }

  void reset()
  {
    derNm1_ = 0.;
    init_ = false;
  }

  void reset(const T & val, const typename timer_t::time_point tNow)
  {
    timer_.tic(tNow);
    valNm1_ = val;
    derNm1_ = 0.;
    init_ = true;
  }

  void reset(const T & val)
  {
    reset(val, timer_.now());
  }

protected:
  timer_t timer_ = timer_t(clock_t{});
  DerivatorParams prm_;
  double valNm1_ = 0.;
  double derNm1_ = 0.;
  bool init_{false};
};

}  // namespace cbr

// cppcheck-suppress unknownMacro
BOOST_HANA_ADAPT_STRUCT(
  cbr::DerivatorParams,
  filter_tau
);

#endif  // CBR_CONTROL__DERIVATOR_HPP_

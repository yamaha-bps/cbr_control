// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__FIRST_ORDER_FILTER_HPP_
#define CBR_CONTROL__FIRST_ORDER_FILTER_HPP_

#include <Eigen/Core>

#include <cbr_utils/cyber_timer.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>


namespace cbr
{

namespace fof_details
{

template<typename T>
struct zero_value
{
  static T value()
  {
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<T>, T>) {
      return T::Zero();
    } else {
      return T{0};
    }
  }
};

}  // namespace fof_details

template<typename T, typename _clock_t = std::chrono::high_resolution_clock>
class FirstOrderFilter
{
public:
  using clock_t = _clock_t;
  using timer_t = CyberTimerNoAvg<std::ratio<1>, double, clock_t>;

  FirstOrderFilter() = default;
  FirstOrderFilter(const FirstOrderFilter &) = default;
  FirstOrderFilter(FirstOrderFilter &) = default;
  FirstOrderFilter(FirstOrderFilter &&) = default;
  FirstOrderFilter & operator=(const FirstOrderFilter &) = default;
  FirstOrderFilter & operator=(FirstOrderFilter &) = default;
  FirstOrderFilter & operator=(FirstOrderFilter &&) = default;
  ~FirstOrderFilter() = default;

  template<typename T1>
  explicit FirstOrderFilter(T1 && clock, const double tau = 1.)
  : timer_(std::forward<T1>(clock)),
    tau_(tau)
  {}

  explicit FirstOrderFilter(const double tau)
  : tau_(tau)
  {}


  template<typename T1>
  void set_clock(T1 && clock)
  {
    timer_.set_clock(std::forward<T1>(clock));
  }

  void set_params(const double tau)
  {
    tau_ = tau;
  }

  double get_params() const
  {
    return tau_;
  }

  const T & update(const T & val, const typename timer_t::time_point tNow)
  {
    if (tau_ <= 0.0) {
      return valNm1_ = val;
    }

    if (init_) {
      const double dt = timer_.toctic(tNow);
      const double e = std::exp(-dt / tau_);
      return valNm1_ = valNm1_ * e + (1 - e) * val;
    }

    init_ = true;
    timer_.tic(tNow);
    return valNm1_ = val;
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
    return valNm1_;
  }

  void reset()
  {
    init_ = false;
  }

  const T & reset(const T & val, const typename timer_t::time_point tNow)
  {
    timer_.tic(tNow);
    init_ = true;
    return valNm1_ = val;
  }

  const T & reset(const T & val)
  {
    return reset(val, timer_.now());
  }

protected:
  timer_t timer_ = timer_t(clock_t{});
  double tau_{1.0};
  T valNm1_{fof_details::zero_value<T>::value()};
  bool init_{false};
};

}  // namespace cbr

#endif  // CBR_CONTROL__FIRST_ORDER_FILTER_HPP_

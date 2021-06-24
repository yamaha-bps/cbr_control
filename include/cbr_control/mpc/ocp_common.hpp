// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_CONTROL__MPC__OCP_COMMON_HPP_
#define CBR_CONTROL__MPC__OCP_COMMON_HPP_

#include <experimental/type_traits>

#include <cbr_math/eigen_traits.hpp>

#include <cstdint>
#include <type_traits>
#include <fstream>
#include <string>


namespace cbr
{

namespace ocp_detail
{
// Discrete
template<typename T>
using has_E_discrete = decltype(std::declval<T &>().get_E(std::size_t {}));

template<typename T>
using has_q_discrete = decltype(std::declval<T &>().get_q(std::size_t {}));

template<typename T>
using has_qT_discrete = decltype(std::declval<T &>().get_qT());

template<typename T>
using has_r_discrete = decltype(std::declval<T &>().get_r(std::size_t {}));

// Continuous
template<typename T>
using has_E_continuous = decltype(std::declval<T &>().get_E(double {}));

template<typename T>
using has_q_continuous = decltype(std::declval<T &>().get_q(double {}));

template<typename T>
using has_qT_continuous = decltype(std::declval<T &>().get_qT());

template<typename T>
using has_r_continuous = decltype(std::declval<T &>().get_r(double {}));


}  // namespace ocp_detail

}  // namespace cbr

#endif  // CBR_CONTROL__MPC__OCP_COMMON_HPP_

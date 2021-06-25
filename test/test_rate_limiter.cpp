// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <gtest/gtest.h>
#include <cbr_control/rate_limiter.hpp>

template class cbr::RateLimiter<std::chrono::high_resolution_clock>;

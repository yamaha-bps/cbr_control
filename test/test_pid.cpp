// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#include <gtest/gtest.h>
#include <cbr_control/pid.hpp>

template class cbr::PID<std::chrono::high_resolution_clock>;

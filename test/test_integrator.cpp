// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#include <gtest/gtest.h>
#include <cbr_control/integrator.hpp>

template class cbr::Integrator<std::chrono::high_resolution_clock>;

// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__PGE__CERES_MARGINALIZATION_HPP_
#define CBR_CONTROL__PGE__CERES_MARGINALIZATION_HPP_

#include <Eigen/Dense>

#include <ceres/problem.h>

#include <cbr_utils/utils.hpp>

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


namespace cbr
{

/** Struct to keep track of variables inside a marginalization factor */
struct MarginalizationInfo
{
  MarginalizationInfo(const double * p, int idx0, int size, int gsize, const Eigen::VectorXd & mu)
  : p(p), idx0(idx0), size(size), gsize(gsize), mu(mu) {}
  const double * p;     // variable pointer
  int idx0;             // start index in marginalization factor
  int size;             // local size (tangent space)
  int gsize;            // global size (manifold)
  Eigen::VectorXd mu;   // nominal value in marginalization factor
};


/**
 * Residual for marginalization factor
 * ONLY WORKS WHEN ALL VARIABLES IN VAR_INFO USE IdentityParameterization,
 * in general xi ominus mu_i = Log(x1 * mu_1.inverse).log() should be used.
 *
 *
 *                         [ Log(x1 - mu_1) - gamma1 ]
 * r({ x_i }) = sqrt_inf * [         ...             ]
 *                         [ Log(xk - mu_k) - gammak ]
 *
 *
 * @param gamma offset vector
 * @param sqrt_inf square root information matrix
 * @param var_info variable ordering information
 */
class MarginalizationCost
{
public:
  MarginalizationCost(
    const Eigen::VectorXd & gamma, const Eigen::MatrixXd & sqrt_inf,
    const std::vector<MarginalizationInfo> & var_info)
  : gamma_(gamma), sqrt_inf_(sqrt_inf), var_info_(var_info) {}

  template<typename T>
  bool operator()(T const * const * p_ptr, T * res_ptr) const
  {
    Eigen::Map<Eigen::Matrix<T, -1, 1>> res(res_ptr, sqrt_inf_.rows());
    Eigen::Matrix<T, -1, 1> rhs(gamma_.size());

    for (size_t i = 0; i != var_info_.size(); ++i) {
      const MarginalizationInfo & var = var_info_[i];

      Eigen::Map<const Eigen::Matrix<T, -1, 1>> p(p_ptr[i], var.gsize);
      rhs.segment(var.idx0, var.size) = p - var.mu;
    }

    res = sqrt_inf_ * (rhs - gamma_);
    return true;
  }

private:
  Eigen::VectorXd gamma_;
  Eigen::MatrixXd sqrt_inf_;
  std::vector<MarginalizationInfo> var_info_;
};


/**
 * Compute marginalization factor for a parameter block from a ceres problem
 *
 * Returns gamma, sqrt_inf such that the marginalized nodes can be replace by
 * a residual factor
 *
 *                         [ Log(x1 * mu_1^{-1}) - gamma1 ]
 * r({ x_i }) = sqrt_inf * [              ...             ]
 *                         [ Log(xk * mu_k^{-1}) - gammak ]
 *
 * where xi are all neighbors of the marginalized node. The output variable
 * varinfo contains information about the neighbors xi.
 *
 * @param[in] problem the problem
 * @param[in] node pointer to the parameter to marginalize
 * @param[out] varinfo information about the variables in the marginalizing factor
 * @param[out] gamma offset vector for marginalizing factor
 * @param[out] sqrt_inf square root information matrix for marginalizing factor
 * @return true if marginalization succeeded
 */
bool marginalize(
  const ceres::Problem * problem, const double * node,
  std::vector<MarginalizationInfo> & varinfo,
  Eigen::VectorXd & gamma, Eigen::MatrixXd & sqrt_inf);

}  // namespace cbr

#endif  // CBR_CONTROL__PGE__CERES_MARGINALIZATION_HPP_

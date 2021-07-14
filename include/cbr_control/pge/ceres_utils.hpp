// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_control/blob/master/LICENSE

#ifndef CBR_CONTROL__PGE__CERES_UTILS_HPP_
#define CBR_CONTROL__PGE__CERES_UTILS_HPP_

#include <Eigen/Dense>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <ceres/local_parameterization.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/sized_cost_function.h>

#include <cbr_utils/utils.hpp>

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


namespace cbr
{

/**
 * Ceres local parameterization LieGroup
 */
template<typename LieGroup>
class LieGroupParameterization : public ceres::LocalParameterization
{
public:
  using JacT = Eigen::Matrix<double, LieGroup::num_parameters, LieGroup::DoF, Eigen::RowMajor>;

  bool Plus(const double * x, const double * delta, double * x_plus_delta) const override
  {
    const Eigen::Map<const LieGroup> x_w(x);
    const Eigen::Map<const typename LieGroup::Tangent> delta_w(delta);
    Eigen::Map<LieGroup> x_plus_delta_w(x_plus_delta);
    x_plus_delta_w = x_w * LieGroup::exp(delta_w);
    return true;
  }

  bool ComputeJacobian(const double * x, double * jacobian) const override
  {
    const Eigen::Map<const LieGroup> x_w(x);
    Eigen::Map<JacT> jacobian_w(jacobian);
    jacobian_w = x_w.Dx_this_mul_exp_x_at_0();
    return true;
  }

  int GlobalSize() const override
  {
    return LieGroup::num_parameters;
  }

  int LocalSize() const override
  {
    return LieGroup::DoF;
  }
};


namespace detail
{

template<typename T>
struct lie_group_dimensions;

template<typename ... Ts>
struct lie_group_dimensions<cbr::TypePack<Ts...>>
{
  using type = std::integer_sequence<int, Ts::num_parameters ...>;
};


template<typename T>
struct strip_map
{
  using type = T;
};

template<typename T>
struct strip_map<Eigen::Map<T>>
{
  using type = T;
};

template<typename T>
struct strip_map<Eigen::Map<const T>>
{
  using type = T;
};

template<typename T>
using strip_map_t = typename strip_map<T>::type;

// deduce cost function arguments
template<typename Cost, typename Scalar>
using cost_param_t =
  typename cbr::signature<decltype(&Cost::template operator()<Scalar>)>::argument_type
  ::decay::template apply<strip_map_t>;

// deduce cost function return type
template<typename Cost, typename Scalar>
using cost_residual_t =
  typename signature<decltype(&Cost::template operator()<Scalar>)>::return_type;

// deduce cost function problem dimensions (NRes, N0, N1, ...)
template<typename Cost>
using cost_dims_t = iseq_join_t<
  std::integer_sequence<int, cost_residual_t<Cost, double>::SizeAtCompileTime>,
  typename lie_group_dimensions<cost_param_t<Cost, double>>::type
>;

}  // namespace detail

/**
  * \brief ceres::SizedCostFunction functor around a Cost implementation using autodiff automatic differentiation
 *
 * If a Cost has a member R = operator()<T>(const S1 &, const S2 &) templated on the Scalar type
 *
 * where
 *   - Si = std::tuple<Si1, Si2...> is a tuple corresponding to a LieState
 *   - Ri is an Eigen::Matrix<Scalar, num_res, 1> of residuals
 *
 * then SizedCostFunction<Cost>::create(std::unique_ptr<Cost>) creates a valid ceres cost function.
 *
 * This seems to call the cost function more than the built-in ceres method
 */
template<typename Cost>
class AutodiffCostWrapper : public iseq_apply_t<detail::cost_dims_t<Cost>, ceres::SizedCostFunction>
{
  static constexpr size_t RSize = detail::cost_residual_t<Cost, double>::SizeAtCompileTime;

  using idx_t = std::make_index_sequence<detail::cost_param_t<Cost, double>::size>;

  using len_t = typename detail::lie_group_dimensions<detail::cost_param_t<Cost, double>>::type;
  using beg_t = cbr::iseq_psum_t<len_t>;

public:
  static ceres::CostFunction * create(std::unique_ptr<Cost> cost)
  {
    return new AutodiffCostWrapper(std::move(cost));
  }

  // Interface required by ceres::SizedCostFunction
  bool Evaluate(const double * const * prm, double * res, double ** jac) const override
  {
    return eval_(prm, res, jac, idx_t{}, beg_t{}, len_t{});
  }

private:
  explicit AutodiffCostWrapper(std::unique_ptr<Cost> cost)
  : cost_{std::move(cost)}
  {}

  // virtual ~AutodiffCostWrapper() = default;

  template<size_t ... Idx, int ... Beg, int ... Len>
  bool eval_(
    const double * const * prm, double * res, double ** jac,
    const std::index_sequence<Idx...>, const std::integer_sequence<int, Beg...>,
    const std::integer_sequence<int, Len...>) const
  {
    // wrap residual in Eigen::Map
    Eigen::Map<detail::cost_residual_t<Cost, double>> res_wrap(res);

    if (jac) {
      // Cost function with inputs/outputs as fixed-size eigen structures
      const auto fcn =
        [this](const Eigen::Matrix<autodiff::dual, Len, 1> & ... args)
        -> Eigen::Matrix<autodiff::dual, RSize, 1>
        {
          return cost_->template operator()<autodiff::dual>(
            Eigen::Map<
              const typename detail::cost_param_t<Cost, autodiff::dual>::template type<Idx>
            >(args.data())
            ...
          );
        };

      // copy parameters into tuple of Eigen::Vector<autodiff::dual>
      auto prm_wrap_autodiff = std::make_tuple(
        Eigen::Matrix<autodiff::dual, Len,
        1>(Eigen::Map<const Eigen::Matrix<double, Len, 1>>(prm[Idx]))...
      );

      // calculate residuals and derivatives
      Eigen::Matrix<autodiff::dual, RSize, 1> F;
      Eigen::Matrix<double, RSize, (Len + ...)> J = autodiff::forward::jacobian(
        fcn, autodiff::forward::wrtpack(std::get<Idx>(prm_wrap_autodiff) ...),
        autodiff::forward::at(std::get<Idx>(prm_wrap_autodiff) ...), F);

      // wrap jac in Eigen::Matrix<autodiff::dual>
      auto jac_wrap = std::make_tuple(
        Eigen::Map<Eigen::Matrix<double, RSize, Len, Eigen::RowMajor>>(jac[Idx]) ...
      );

      // write function value into residual wrapper
      res_wrap = F.template cast<double>();

      // write jacobian blocks into wrapper
      (std::get<Idx>(jac_wrap).operator=(J.template block<RSize, Len>(0, Beg)), ...);
    } else {
      res_wrap = cost_->template operator()<double>(
        Eigen::Map<const typename detail::cost_param_t<Cost, double>::template type<Idx>>(
          prm[Idx]) ...
      );
    }

    return true;
  }

  std::unique_ptr<Cost> cost_;
};


/**
 * \brief ceres::AutodiffCostFunction functor around a Cost implementation
 *
 * If a Cost has a member R = operator()<T>(const S1 &, const S2 &) templated on the Scalar type
 *
 * where
 *   - Si = std::tuple<Si1, Si2...> is a tuple corresponding to a LieState
 *   - Ri is an Eigen::Matrix<Scalar, num_res, 1> of residuals
 *
 * then CeresCostFunctor<Cost>::create(std::unique_ptr<Cost>) creates a valid ceres cost function.
 */
template<typename Cost>
class CeresCostFunctor
{
public:
  static ceres::CostFunction * create(std::unique_ptr<Cost> cost)
  {
    using CostFcn = iseq_apply_t<detail::cost_dims_t<Cost>, CostFcnNoDims>;
    return new CostFcn(new CeresCostFunctor<Cost>(std::move(cost)));
  }

  // Interface required by ceres::AutoDiffCostFunction
  template<typename ... Args>
  bool operator()(Args && ... args) const
  {
    return eval_(std::make_index_sequence<sizeof...(args) - 1>{}, std::forward<Args>(args) ...);
  }

private:
  template<int ... Idx>
  using CostFcnNoDims = ceres::AutoDiffCostFunction<CeresCostFunctor<Cost>, Idx...>;

  explicit CeresCostFunctor(std::unique_ptr<Cost> cost)
  : cost_{std::move(cost)}
  {}

  template<typename ... Args, size_t ... Idx>
  bool eval_(const std::index_sequence<Idx...>, Args ... args) const
  {
    auto arg_tuple = std::make_tuple(args ...);

    // split arg_tuple into parameters and residual
    auto prm = cbr::sub_tuple(arg_tuple, std::index_sequence<Idx...>{});
    auto res = cbr::sub_tuple(arg_tuple, std::index_sequence<sizeof...(Idx)>{});

    // deduce scalar type
    using scalar_t = std::remove_cv_t<std::remove_pointer_t<
          std::remove_reference_t<decltype(std::get<0>(res))>>>;

    // wrap the residual
    Eigen::Map<detail::cost_residual_t<Cost, scalar_t>> res_wrap(std::get<0>(res));

    // wrap the parameters and call cost function
    res_wrap = cost_->template operator()<scalar_t>(
      Eigen::Map<const typename detail::cost_param_t<Cost, scalar_t>::template type<Idx>>(
        std::get<Idx>(prm)
      ) ...);

    return true;
  }

  std::unique_ptr<Cost> cost_;
};

}  // namespace cbr

#endif  // CBR_CONTROL__PGE__CERES_UTILS_HPP_

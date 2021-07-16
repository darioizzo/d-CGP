#ifndef DCGP_WRAPPED_FUNCTIONS_H
#define DCGP_WRAPPED_FUNCTIONS_H

#include <audi/audi.hpp>
#include <audi/functions.hpp>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <dcgp/config.hpp>
#include <dcgp/function.hpp>
#include <dcgp/type_traits.hpp>

#define DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(f)                                                                         \
    DCGP_S11N_FUNCTION_EXPORT_KEY(dcgp_##f##_double, dcgp::f##_func<double>, double, const std::vector<double> &)      \
    DCGP_S11N_FUNCTION_EXPORT_KEY(dcgp_##f##_gdual_d, dcgp::f##_func<audi::gdual_d>, audi::gdual_d,                    \
                                  const std::vector<audi::gdual_d> &)                                                  \
    DCGP_S11N_FUNCTION_EXPORT_KEY(dcgp_##f##_gdual_v, dcgp::f##_func<audi::gdual_v>, audi::gdual_v,                    \
                                  const std::vector<audi::gdual_v> &)

#define DCGP_S11N_FUNCTION_IMPLEMENT_MULTI(f)                                                                          \
    DCGP_S11N_FUNCTION_IMPLEMENT(dcgp_##f##_double, dcgp::f##_func<double>, double, const std::vector<double> &)       \
    DCGP_S11N_FUNCTION_IMPLEMENT(dcgp_##f##_gdual_d, dcgp::f##_func<audi::gdual_d>, audi::gdual_d,                     \
                                 const std::vector<audi::gdual_d> &)                                                   \
    DCGP_S11N_FUNCTION_IMPLEMENT(dcgp_##f##_gdual_v, dcgp::f##_func<audi::gdual_v>, audi::gdual_v,                     \
                                 const std::vector<audi::gdual_v> &)

#define DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(f)                                                                        \
    DCGP_S11N_FUNCTION_EXPORT_KEY(dcgp_##f##_string, dcgp::f##_func, std::string, const std::vector<std::string> &)

#define DCGP_S11N_FUNCTION_IMPLEMENT_STRING(f)                                                                         \
    DCGP_S11N_FUNCTION_IMPLEMENT(dcgp_##f##_string, dcgp::f##_func, std::string, const std::vector<std::string> &)

#define DCGP_S11N_EMPTY_SERIALIZE_MEMFN()                                                                              \
    template <typename Archive>                                                                                        \
    void serialize(Archive &, unsigned)                                                                                \
    {                                                                                                                  \
    }

namespace dcgp
{

// SFINAE dust (to hide under the carpet). Its used to enable the templated
// version of the various functions that can construct a kernel object. Only for
// double and a gdual type Complex could also be allowed.
template <typename T>
using f_enabler = typename std::enable_if<std::is_same<T, double>::value || is_gdual<T>::value, int>::type;

/*--------------------------------------------------------------------------
 *                              N-ARITY FUNCTIONS
 *------------------------------------------------------------------------**/

template <typename T, f_enabler<T> = 0>
struct my_diff_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval -= in[i];
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_diff = my_diff_func<T>{};

struct print_my_diff_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "-" + in[i];
        }
        return "(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_diff = print_my_diff_func{};

template <typename T, f_enabler<T> = 0>
struct my_mul_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval *= in[i];
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_mul = my_mul_func<T>{};

struct print_my_mul_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "*" + in[i];
        }
        return "(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_mul = print_my_mul_func{};

template <typename T, f_enabler<T> = 0>
struct my_div_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval /= in[i];
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_div = my_div_func<T>{};

struct print_my_div_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "/" + in[i];
        }
        return "(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_div = print_my_div_func{};

// Protected divide function (double overload):
template <typename T, typename = void>
struct my_pdiv_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        T tmpval(in[1]);

        for (auto i = 2u; i < in.size(); ++i) {
            tmpval *= in[i];
        }

        retval /= tmpval;

        if (std::isfinite(retval)) {
            return retval;
        }

        return 1.;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

// Protected divide function (gdual overload):
// this will throw when used.
// The pdiv is only available as a double type, for use in CGP.
// Because the gradients created when using gdual are mathematically invalid.
template <typename T>
struct my_pdiv_func<T, std::enable_if_t<is_gdual<T>::value>> {
    /// Call operator
    T operator()(const std::vector<T> &) const
    {
        throw std::invalid_argument("The protected division is not supported for gdual types.");
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_pdiv = my_pdiv_func<T>{};

struct print_my_pdiv_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "/" + in[i];
        }

        return "(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_pdiv = print_my_pdiv_func{};

/*--------------------------------------------------------------------------
 *                            Suitable for dCGPANN
 *------------------------------------------------------------------------**/

// sigmoid function: 1 / (1 + exp(- (a + b + c + d+ .. + ))
template <typename T, f_enabler<T> = 0>
struct my_sig_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return 1. / (1. + audi::exp(-retval));
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_sig = my_sig_func<T>{};

struct print_my_sig_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "sig(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_sig = print_my_sig_func{};

// tanh function:
template <typename T, f_enabler<T> = 0>
struct my_tanh_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return audi::tanh(retval);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_tanh = my_tanh_func<T>{};

struct print_my_tanh_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "tanh(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_tanh = print_my_tanh_func{};

// ReLu function (double overload):
template <typename T, typename = void>
struct my_relu_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        if (retval < 0) {
            retval = T(0.);
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

// ReLu function (gdual overload):
template <typename T>
struct my_relu_func<T, std::enable_if_t<is_gdual<T>::value>> {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        if (retval.constant_cf() < T(0.).constant_cf()) {
            retval = T(0.);
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_relu = my_relu_func<T>{};

struct print_my_relu_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "ReLu(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_relu = print_my_relu_func{};

// Exponential linear unit (ELU) function (double overload):
template <typename T, typename = void>
struct my_elu_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        if (retval < 0) {
            retval = audi::exp(retval) - T(1.);
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

// Exponential linear unit (ELU) function (gdual overload):
template <typename T>
struct my_elu_func<T, std::enable_if_t<is_gdual<T>::value>> {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        if (retval.constant_cf() < T(0.).constant_cf()) {
            retval = audi::exp(retval) - T(1.);
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_elu = my_elu_func<T>{};

struct print_my_elu_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "ELU(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_elu = print_my_elu_func{};

// Inverse square root function: x / sqrt(1+x^2):
template <typename T, f_enabler<T> = 0>
struct my_isru_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return retval / (audi::sqrt(1 + retval * retval));
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_isru = my_isru_func<T>{};

struct print_my_isru_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "ISRU(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_isru = print_my_isru_func{};

template <typename T, f_enabler<T> = 0>
struct my_sum_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_sum = my_sum_func<T>{};

struct print_my_sum_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_sum = print_my_sum_func{};

// sine function (non unary version):
template <typename T, f_enabler<T> = 0>
struct my_sin_nu_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return audi::sin(retval);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_sin_nu = my_sin_nu_func<T>{};

struct print_my_sin_nu_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "sin(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_sin_nu = print_my_sin_nu_func{};

// cosine function (non unary version):
template <typename T, f_enabler<T> = 0>
struct my_cos_nu_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return audi::cos(retval);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_cos_nu = my_cos_nu_func<T>{};

struct print_my_cos_nu_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "cos(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_cos_nu = print_my_cos_nu_func{};

// gaussian function (non unary version):
template <typename T, f_enabler<T> = 0>
struct my_gaussian_nu_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return audi::exp(-retval*retval);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_gaussian_nu = my_gaussian_nu_func<T>{};

struct print_my_gaussian_nu_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "exp(-(" + retval + ")**2)";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_gaussian_nu = print_my_gaussian_nu_func{};

// inv sum function:
template <typename T, f_enabler<T> = 0>
struct my_inv_sum_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return -retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_inv_sum = my_inv_sum_func<T>{};

struct print_my_inv_sum_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "-(" + retval +")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_inv_sum = print_my_inv_sum_func{};

// abs function:
template <typename T, f_enabler<T> = 0>
struct my_abs_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return audi::abs(retval);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_abs = my_abs_func<T>{};

struct print_my_abs_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "abs(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_abs = print_my_abs_func{};

// step function:
template <typename T, typename = void>
struct my_step_func {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        return retval < T(0.) ? T(0.) : T(1.);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

// step function (gdual overload):
template <typename T>
struct my_step_func<T, std::enable_if_t<is_gdual<T>::value>> {
    T operator()(const std::vector<T> &in) const
    {
        T retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += in[i];
        }
        if (retval.constant_cf() < T(0.).constant_cf()) {
            retval = T(0.);
        } else {
            retval = T(1.);
        }
        return retval;
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_step = my_step_func<T>{};

struct print_my_step_func {
    std::string operator()(const std::vector<std::string> &in) const
    {
        std::string retval(in[0]);
        for (auto i = 1u; i < in.size(); ++i) {
            retval += "+" + in[i];
        }
        return "step(" + retval + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_step = print_my_step_func{};

/*--------------------------------------------------------------------------
 *                               UNARY FUNCTIONS
 *------------------------------------------------------------------------**/
// sine
template <typename T, f_enabler<T> = 0>
struct my_sin_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return sin(in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_sin = my_sin_func<T>{};

struct print_my_sin_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "sin(" + in[0] + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_sin = print_my_sin_func{};

// cosine
template <typename T, f_enabler<T> = 0>
struct my_cos_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return cos(in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_cos = my_cos_func<T>{};

struct print_my_cos_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "cos(" + in[0] + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_cos = print_my_cos_func{};

// logarithm
template <typename T, f_enabler<T> = 0>
struct my_log_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return audi::log(in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_log = my_log_func<T>{};

struct print_my_log_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "log(" + in[0] + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_log = print_my_log_func{};

// exponential (unary)
// This exponential discards all inputs except the first one
template <typename T, f_enabler<T> = 0>
struct my_exp_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return audi::exp(in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_exp = my_exp_func<T>{};

struct print_my_exp_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "exp(" + in[0] + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_exp = print_my_exp_func{};

// gaussian (unary)
// This gaussian discards all inputs except the first one
template <typename T, f_enabler<T> = 0>
struct my_gaussian_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return audi::exp(-in[0] * in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_gaussian = my_gaussian_func<T>{};

struct print_my_gaussian_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "exp(-" + in[0] + "**2)";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_gaussian = print_my_gaussian_func{};

// sqrt (unary)
// This square root discards all inputs except the first one
template <typename T, f_enabler<T> = 0>
struct my_sqrt_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return audi::sqrt(in[0]);
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_sqrt = my_sqrt_func<T>{};

struct print_my_sqrt_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "sqrt(" + in[0] + ")";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_sqrt = print_my_sqrt_func{};

// protected sqrt (unary)
// This protected square root discards all inputs except the first one
template <typename T, f_enabler<T> = 0>
struct my_psqrt_func {
    /// Call operator
    T operator()(const std::vector<T> &in) const
    {
        return audi::sqrt(audi::abs(in[0]));
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

template <typename T>
inline constexpr auto my_psqrt = my_psqrt_func<T>{};

struct print_my_psqrt_func {
    /// Call operator
    std::string operator()(const std::vector<std::string> &in) const
    {
        return "sqrt(abs(" + in[0] + "))";
    }
    DCGP_S11N_EMPTY_SERIALIZE_MEMFN()
};

inline constexpr auto print_my_psqrt = print_my_psqrt_func{};

} // namespace dcgp

DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_diff)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_diff)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_sqrt)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_sqrt)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_psqrt)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_psqrt)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_mul)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_mul)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_div)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_div)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_pdiv)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_pdiv)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_sig)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_sig)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_tanh)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_tanh)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_relu)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_relu)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_elu)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_elu)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_isru)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_isru)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_sum)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_sum)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_sin)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_sin)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_cos)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_cos)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_log)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_log)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_exp)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_exp)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_gaussian)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_gaussian)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_sin_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_sin_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_cos_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_cos_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_gaussian_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_gaussian_nu)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_inv_sum)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_inv_sum)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_abs)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_abs)
DCGP_S11N_FUNCTION_EXPORT_KEY_MULTI(my_step)
DCGP_S11N_FUNCTION_EXPORT_KEY_STRING(print_my_step)

#endif // DCGP_WRAPPED_FUNCTIONS_H

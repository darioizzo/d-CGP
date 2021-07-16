#ifndef DCGP_KERNEL_H
#define DCGP_KERNEL_H

#include <algorithm>
#include <iostream>
#include <string>
#include <utility> // std::forward
#include <vector>

#include <dcgp/config.hpp>
#include <dcgp/function.hpp>
#include <dcgp/s11n.hpp>

namespace dcgp
{

/// Basis function
/**
 * This class represents the function defining the generic CGP node. To be constructed
 * it accepts two functions having prototype ``T``(const std::vector<``T``>&) and std::string(const
 * std::vector<std::string>&) computing, respectively, the function value on generic inputs and the textual
 * representation of the operation.
 *
 * The intended use, for an example with ``T`` = ``double`` would then be something like:
 * @code
 * inline double my_sum(const std::vector<double> &in)
 * {
 *     T retval(in[0]);
 *     for (auto i = 1u; i < in.size(); ++i) {
 *         retval += in[i];
 *     }
 *     return retval;
 * }
 *
 * inline std::string print_my_sum(const std::vector<std::string> &in)
 * {
 *     std::string retval(in[0]);
 *     for (auto i = 1u; i < in.size(); ++i) {
 *         retval += "+" + in[i];
 *     }
 *     return "(" + retval + ")";
 * }
 *
 * kernel<double> f(my_sum<double>, print_my_sum, "my_sum");
 * @endcode
 *
 * @tparam T The type of the function output (and inputs)
 */
template <typename T>
class kernel
{
public:
#if !defined(DCGP_DOXYGEN_INVOKED)
    /// Basic prototype of a kernel function returning its evaluation
    using my_fun_type = function<T(const std::vector<T> &)>;
    /// Basic prototype of a kernel function returning its symbolic representation
    using my_print_fun_type = function<std::string(const std::vector<std::string> &)>;
#endif
    kernel() = default;

    /// Constructor
    /**
     * Constructs a kernel that can be used as kernel in a dCGP expression
     *
     * @param[in] f any callable with prototype T(const std::vector<T>&)
     * @param[in] pf any callable with prototype std::string(const std::vector<std::string>&)
     * @param[in] name string containing the function name (ex. "sum")
     *
     */
    template <typename U, typename V>
    kernel(U &&f, V &&pf, std::string name) : m_f(std::forward<U>(f)), m_pf(std::forward<V>(pf)), m_name(name)
    {
        m_thread_safety = std::min(m_f.get_thread_safety(), m_pf.get_thread_safety());
    }

    /// Parenthesis operator
    /**
     * Evaluates the kernel in the point \p in
     *
     * @param[in] in the evaluation point as an std::vector<T>
     *
     * @return the function value
     */
    T operator()(const std::vector<T> &in) const
    {
        return m_f(in);
    }
    /// Parenthesis operator
    /**
     * Evaluates the kernel in the point \p in
     *
     * @param[in] in the evaluation point as an std::initializer_list<T>
     *
     * @return the function value
     */
    T operator()(const std::initializer_list<T> &in) const
    {
        return m_f(in);
    }
    /// Parenthesis operator
    /**
     * Returns a symbolic representation of the operation made by \f$f\f$
     *
     * @param[in] in std::vector<std::string> with the symbolic names to be used
     *
     * @return the string representation of the operation (ex. "ln(s1+s2)")
     */
    std::string operator()(const std::vector<std::string> &in) const
    {
        return m_pf(in);
    }

    /// Kernel name
    /**
     * Returns the Kernel name
     *
     * @return the Kernel name
     */
    const std::string &get_name() const
    {
        return m_name;
    }

    // Thread safety level.
    pagmo::thread_safety get_thread_safety() const
    {
        return m_thread_safety;
    }

    /// Overloaded stream operator
    /**
     * Will stream the function name
     *
     * @param[in,out] os target stream.
     * @param[in] d kernel argument.
     *
     * @return reference to \p os.
     *
     */
    friend std::ostream &operator<<(std::ostream &os, const kernel<T> &d)
    {
        os << d.m_name;
        return os;
    }

    /// Object serialization
    /**
     * This method will save/load \p this into the archive \p ar.
     *
     * @param ar target archive.
     *
     * @throws unspecified any exception thrown by the serialization of the expression and of primitive types.
     */
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_f;
        ar &m_pf;
        ar &m_name;
    }

private:
    /// The function
    my_fun_type m_f;
    /// Its symbolic representation
    my_print_fun_type m_pf;
    /// Its name
    std::string m_name;
    // Thread safety.
    pagmo::thread_safety m_thread_safety;
};

} // end of namespace dcgp

#endif // DCGP_KERNEL_H

#ifndef DCGP_kernel_set_H
#define DCGP_kernel_set_H

#include <audi/audi.hpp>
#include <type_traits>
#include <vector>

#include <dcgp/config.hpp>
#include <dcgp/kernel.hpp>
#include <dcgp/s11n.hpp>
#include <dcgp/wrapped_functions.hpp>

namespace dcgp
{

/// Function set
/**
 * This class is provided as an helper to construct the std::vector<kernel<T>>
 * that is requested to form a dcgp::expression<T>. Once constructed, a call to its
 * parenthesis operator will return the std::vector containing the requested kernels
 *
 * @tparam T The type of the functions output (and inputs)
 */
template <typename T>
class kernel_set
{
public:
    /// Constructor
    /**
     * Default constructor
     */
    kernel_set() : m_kernels(){};

    /// Constructor
    /**
     * Constructs a kernel set that can be used in a dCGP expression
     *
     * @param[in] list an std::vector of strings containing the function names (e.g., "sum")
     */
    kernel_set(const std::vector<std::string> &list)
    {
        for (auto kernel_name : list) {
            push_back(kernel_name);
        }
    }

    /// Adds a kernel to the set
    /**
     * Constructs a kernel<T> given a string containing the function name, and
     * inserts it into the std::vector
     *
     * @param[in] kernel_name a string containing the function name
     *
     * @throw std::invalid_argument if the function is not implemented
     */
    void push_back(std::string kernel_name)
    {
        if (kernel_name == "sum")
            m_kernels.emplace_back(my_sum<T>, print_my_sum, kernel_name);
        else if (kernel_name == "diff")
            m_kernels.emplace_back(my_diff<T>, print_my_diff, kernel_name);
        else if (kernel_name == "mul")
            m_kernels.emplace_back(my_mul<T>, print_my_mul, kernel_name);
        else if (kernel_name == "div")
            m_kernels.emplace_back(my_div<T>, print_my_div, kernel_name);
        //  pdiv is only available when class type is double
        else if (kernel_name == "pdiv" && std::is_same<T, double>::value)
            m_kernels.emplace_back(my_pdiv<T>, print_my_pdiv, kernel_name);
        else if (kernel_name == "sig")
            m_kernels.emplace_back(my_sig<T>, print_my_sig, kernel_name);
        else if (kernel_name == "tanh")
            m_kernels.emplace_back(my_tanh<T>, print_my_tanh, kernel_name);
        else if (kernel_name == "ReLu")
            m_kernels.emplace_back(my_relu<T>, print_my_relu, kernel_name);
        else if (kernel_name == "ELU")
            m_kernels.emplace_back(my_elu<T>, print_my_elu, kernel_name);
        else if (kernel_name == "ISRU")
            m_kernels.emplace_back(my_isru<T>, print_my_isru, kernel_name);
        else if (kernel_name == "sin")
            m_kernels.emplace_back(my_sin<T>, print_my_sin, kernel_name);
        else if (kernel_name == "cos")
            m_kernels.emplace_back(my_cos<T>, print_my_cos, kernel_name);
        else if (kernel_name == "log")
            m_kernels.emplace_back(my_log<T>, print_my_log, kernel_name);
        else if (kernel_name == "exp")
            m_kernels.emplace_back(my_exp<T>, print_my_exp, kernel_name);
        else if (kernel_name == "gaussian")
            m_kernels.emplace_back(my_gaussian<T>, print_my_gaussian, kernel_name);
        else if (kernel_name == "sqrt")
            m_kernels.emplace_back(my_sqrt<T>, print_my_sqrt, kernel_name);
        else if (kernel_name == "psqrt")
            m_kernels.emplace_back(my_psqrt<T>, print_my_psqrt, kernel_name);
        else if (kernel_name == "sin_nu")
            m_kernels.emplace_back(my_sin_nu<T>, print_my_sin_nu, kernel_name);
        else if (kernel_name == "cos_nu")
            m_kernels.emplace_back(my_cos_nu<T>, print_my_cos_nu, kernel_name);
        else if (kernel_name == "gaussian_nu")
            m_kernels.emplace_back(my_gaussian_nu<T>, print_my_gaussian_nu, kernel_name);
        else if (kernel_name == "inv_sum")
            m_kernels.emplace_back(my_inv_sum<T>, print_my_inv_sum, kernel_name);
        else if (kernel_name == "abs")
            m_kernels.emplace_back(my_abs<T>, print_my_abs, kernel_name);
        else if (kernel_name == "step")
            m_kernels.emplace_back(my_step<T>, print_my_step, kernel_name);
        else
            throw std::invalid_argument("Unimplemented function " + kernel_name + " for this type");
    }

    /// Adds a kernel to the set
    /**
     * Inserts the given kernel<T> into the std::vector
     *
     * @param[in] kernel the dcgp::kernel<T> to add
     */
    void push_back(const dcgp::kernel<T> &kernel)
    {
        m_kernels.push_back(kernel);
    }

    /// Clears the kernel set
    /**
     * Removes all the elements from the std::vector containing the kernels
     */
    void clear()
    {
        m_kernels.clear();
    }

    /// Overloaded function call operator
    /**
     * Returns the std::vector containing the kernels
     */
    std::vector<dcgp::kernel<T>> operator()() const
    {
        return m_kernels;
    }

    /// Overloaded stream operator
    /**
     * Will stream the function names
     *
     * @param[in,out] os target stream
     * @param[in] d kernel_set argument
     *
     * @return reference to \p os
     */
    friend std::ostream &operator<<(std::ostream &os, const kernel_set<T> &d)
    {
        audi::stream(os, d());
        return os;
    }

    // Overloaded subscript operator
    /**
     * Returns the kernel at a specified index in the std::vector
     *
     * @param[in] idx index of the required kernel
     *
     * @return the kernel at the specified index
     */
    dcgp::kernel<T> operator[](const typename std::vector<dcgp::kernel<T>>::size_type idx) const
    {
        return m_kernels[idx];
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
        ar &m_kernels;
    }

private:
    // vector of functions
    std::vector<dcgp::kernel<T>> m_kernels;
};

} // end of namespace dcgp

#endif // DCGP_kernel_set_H

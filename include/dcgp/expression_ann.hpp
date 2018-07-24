#ifndef DCGP_EXPRESSION_ANN_H
#define DCGP_EXPRESSION_ANN_H

#include <audi/audi.hpp>
#include <initializer_list>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <dcgp/expression.hpp>
#include <dcgp/kernel.hpp>
#include <dcgp/type_traits.hpp>

namespace dcgp
{

/// A dCGP-ANN expression
/**
 * This class represents an artificial neural network as a differentiable Cartesian Genetic
 * program. It add weights, biases and backward automated differentiation to the class
 * dcgp::expression.
 *
 * @tparam T expression type. Can only be a float type
 *
 * @author Dario Izzo (dario.izzo@gmail.com)
 */
template <typename T>
class expression_ann : public expression<T>
{

private:
    // SFINAE dust enables only string or double
    template <typename U>
    using functor_enabler =
        typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::string>::value, int>::type;

public:
    /// Constructor
    /** Constructs a dCGP expression
     *
     * @param[in] n number of inputs (independent variables)
     * @param[in] m number of outputs (dependent variables)
     * @param[in] r number of rows of the dCGP
     * @param[in] c number of columns of the dCGP
     * @param[in] l number of levels-back allowed for the dCGP
     * @param[in] arity arity of the basis functions
     * @param[in] f function set. An std::vector of dcgp::kernel<expression::type>
     * @param[in] seed seed for the random number generator (initial expression and mutations depend on this)
     */
    expression_ann(unsigned n,               // n. inputs
                   unsigned m,               // n. outputs
                   unsigned r,               // n. rows
                   unsigned c,               // n. columns
                   unsigned l,               // n. levels-back
                   unsigned arity,           // basis functions' arity
                   std::vector<kernel<T>> f, // functions
                   unsigned seed             // seed for the pseudo-random numbers
                   )
        : expression<T>(n, m, r, c, l, arity, f, seed), m_weights(r * c * arity, T(1.)), m_biases(r * c, T(0.))
    {
        for (auto ker : f) {
            if (ker.get_name() != "tanh" && ker.get_name() != "sig" && ker.get_name() != "ReLu") {
                throw std::invalid_argument("Only tanh, sig and ReLu Kernels are valid for dCGP-ANN expressions");
            }
        }
        for (auto i = 0u; i < r * c; ++i) {
            for (auto j = 0u; j < arity; ++j) {
                m_weights_symbols.push_back("w" + std::to_string(i + n) + "_" + std::to_string(j));
            }
        }
        for (auto i = 0u; i < r * c; ++i) {
            m_biases_symbols.push_back("b" + std::to_string(i + n));
        }
        update_data_structures();
    }

    /// Evaluates the mean square error
    /**
     * Returns the mean squared error and its gradient with respect to weights and biases.
     *
     * @param[in] The input data
     * @param[out] The target data (the label)
     * @return the mse, the gradient of the mse w.r.t. all weights (also inactive) and the gradient of the mse w.r.t all
     * biases
     */
    template <typename U, functor_enabler<U> = 0>
    std::tuple<U, std::vector<U>, std::vector<U>> mse(const std::vector<U> &in, const std::vector<U> &out)
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("When computing the mse the input data dimension seemed wrong, it was: "
                                        + std::to_string(in.size())
                                        + " while I expected: " + std::to_string(this->get_n()));
        }
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("When computing the mse the output data dimension seemed wrong, it was: "
                                        + std::to_string(out.size())
                                        + " while I expected: " + std::to_string(this->get_m()));
        }
        U value(U(0.));
        std::vector<U> gweights(m_weights.size(), U(0.));
        std::vector<U> gbiases(m_biases.size(), U(0.));

        // ------------------------------------------ Forward pass----------------------------------------------------
        // All active nodes outputs get computed as well as
        // the activation function derivatives
        std::unordered_map<unsigned, U> node;
        std::unordered_map<unsigned, U> d_node;
        fill_nodes(in, node, d_node);

        // Compute mse and store it
        unsigned j = 0u;
        for (auto i = this->get().size() - this->get_m(); i < this->get().size(); ++i) {
            auto node_idx = this->get()[i];
            value += (node[node_idx] - out[j]) * (node[node_idx] - out[j]);
            j++;
        }

        // ------------------------------------------ Backward pass-----------------------------------------------------
        // 1 - We update d_node on the output nodes and accounting that
        // mse = sum (O_i-\hat O_i)^2
        j = 0u;
        for (auto i = this->get().size() - this->get_m(); i < this->get().size(); ++i) {
            auto node_idx = this->get()[i];
            d_node[node_idx] = d_node[node_idx] * 2 * (node[node_idx] - out[j]);
            j++;
        }
        // 2 - We iterate backward on all the active nodes (except the input nodes)
        // filling up the gradient information at each node for the incoming weights and relative bias
        for (auto it = this->get_active_nodes().rbegin(); it != this->get_active_nodes().rend() - this->get_n(); ++it) {
            // index of the node in the bias vector
            auto b_idx = *it - this->get_n();
            // index of the node in the weight vector
            auto w_idx = this->get_arity() * (*it - this->get_n());
            // index of the node in the chromosome
            auto c_idx = (*it - this->get_n()) * (this->get_arity() + 1);
            // update d_node (only if the node is not connected to an output node, in which case the update has been
            // done above)
            if (std::find(this->get().end() - this->get_m(), this->get().end(), *it) == this->get().end()) {
                U cum = 0.;
                for (auto i = 0u; i < m_connected[*it].size(); ++i) {
                    cum += m_weights[m_connected[*it][i].second] * d_node[m_connected[*it][i].first];
                }
                d_node[*it] *= cum;
            }
            // fill gradients for weights and biases info
            for (auto i = 0u; i < this->get_arity(); ++i) {
                gweights[w_idx + i] = d_node[*it] * node[this->get()[c_idx + 1 + i]];
            }
            gbiases[b_idx] = d_node[*it];
        }
        return std::make_tuple(std::move(value), std::move(gweights), std::move(gbiases));
    }

    /// Evaluates the dCGP-ANN expression
    /**
     * This evaluates the dCGP-ANN expression. According to the template parameter
     * it will compute the value (double) or a symbolic
     * representation (std::string). Any other type will result in a compilation-time
     * error (SFINAE).
     *
     * @param[in] in an std::vector containing the values where the dCGP-ANN expression has
     * to be computed (doubles or strings)
     *
     * @return The value of the function (an std::vector)
     */
    template <typename U, functor_enabler<U> = 0>
    std::vector<U> operator()(const std::vector<U> &in) const
    {
        std::vector<U> retval(this->get_m());
        auto node = fill_nodes(in);
        for (auto i = 0u; i < this->get_m(); ++i) {
            retval[i] = node[this->get()[(this->get_rows() * this->get_cols()) * (this->get_arity() + 1) + i]];
        }
        return retval;
    }

    /// Evaluates the  dCGP-ANN  expression
    /**
     * This evaluates the dCGP-ANN expression. According to the template parameter
     * it will compute the value (double) or a symbolic
     * representation (std::string). Any other type will result in a compilation-time
     * error (SFINAE).
     *
     * @param[in] is an initializer list containing the values where the dCGP expression has
     * to be computed (doubles, or strings)
     *
     * @return The value of the function (an std::vector)
     */
    template <typename U, functor_enabler<U> = 0>
    std::vector<U> operator()(const std::initializer_list<U> &in) const
    {
        std::vector<U> dummy(in);
        return (*this)(dummy);
    }

    /// Overloaded stream operator
    /**
     * Will return a formatted string containing a human readable representation
     * of the class
     *
     * @return std::string containing a human-readable representation of the problem.
     */
    friend std::ostream &operator<<(std::ostream &os, const expression_ann &d)
    {
        audi::stream(os, "d-CGP Expression:\n");
        audi::stream(os, "\tNumber of inputs:\t\t", d.get_n(), '\n');
        audi::stream(os, "\tNumber of outputs:\t\t", d.get_m(), '\n');
        audi::stream(os, "\tNumber of rows:\t\t\t", d.get_rows(), '\n');
        audi::stream(os, "\tNumber of columns:\t\t", d.get_cols(), '\n');
        audi::stream(os, "\tNumber of levels-back allowed:\t", d.get_levels_back(), '\n');
        audi::stream(os, "\tBasis function arity:\t\t", d.get_arity(), '\n');
        audi::stream(os, "\n\tResulting lower bounds:\t", d.get_lb());
        audi::stream(os, "\n\tResulting upper bounds:\t", d.get_ub(), '\n');
        audi::stream(os, "\n\tCurrent expression (encoded):\t", d.get(), '\n');
        audi::stream(os, "\tActive nodes:\t\t\t", d.get_active_nodes(), '\n');
        audi::stream(os, "\tActive genes:\t\t\t", d.get_active_genes(), '\n');
        audi::stream(os, "\n\tFunction set:\t\t\t", d.get_f(), '\n');
        audi::stream(os, "\n\tWeights:\t\t\t", d.m_weights, '\n');
        audi::stream(os, "\tBiases:\t\t\t\t", d.m_biases, '\n');

        return os;
    }

    /**
     * \defgroup Managing Weights and Biases
     */
    /*@{*/

    /// Sets a weight
    /**
     * Sets a connection weight to a new value
     *
     * @param[in] node_id the id of the node whose weight is being set (convention adopted for node numbering
     * http://ppsn2014.ijs.si/files/slides/ppsn2014-tutorial3-miller.pdf)
     * @param[in] input_id the id of the node input (0 for the first one up to arity-1)
     * @param[in] w the new value of the weight
     *
     * @throws std::invalid_argument if the node_id or input_id are not valid
     */
    void set_weight(typename std::vector<T>::size_type node_id, typename std::vector<T>::size_type input_id, const T &w)
    {
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_rows() * this->get_cols()) {
            throw std::invalid_argument("Requested node id does not exist");
        }
        if (input_id >= this->get_arity()) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }
        auto idx = (node_id - this->get_n()) * this->get_arity() + input_id;
        m_weights[idx] = w;
    }

    /// Sets a weight
    /**
     * Sets a connection weight to a new value
     *
     * @param[in] idx index of the weight to be changed.
     *
     * @throws std::invalid_argument if the node_id or input_id are not valid
     */
    void set_weight(typename std::vector<T>::size_type idx, const T &w)
    {
        m_weights[idx] = w;
    }

    /// Sets all weights
    /**
     * Sets all the connection weights at once
     *
     * @param[in] ws an std::vector containing all the weights to set
     *
     * @throws std::invalid_argument if the input vector dimension is not valid (r*c*arity)
     */
    void set_weights(const std::vector<T> &ws)
    {
        if (ws.size() != m_weights.size()) {
            throw std::invalid_argument("The vector of weights has the wrong dimension");
        }
        m_weights = ws;
    }

    /// Gets a weight
    /**
     * Gets the value of a connection weight
     *
     * @param[in] node_id the id of the node (convention adopted for node numbering
     * http://ppsn2014.ijs.si/files/slides/ppsn2014-tutorial3-miller.pdf)
     * @param[in] input_id the id of the node input (0 for the first one up to arity-1)
     *
     * @return the value of the weight
     *
     * @throws std::invalid_argument if the node_id or input_id are not valid
     */
    T get_weight(typename std::vector<T>::size_type node_id, typename std::vector<T>::size_type input_id) const
    {
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_rows() * this->get_cols()) {
            throw std::invalid_argument("Requested node id does not exist");
        }
        if (input_id >= this->get_arity()) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }
        auto idx = (node_id - this->get_n()) * this->get_arity() + input_id;
        return m_weights[idx];
    }

    /// Gets a weight
    /**
     * Gets the value of a connection weight
     *
     * @param[in] idx index of the weight
     *
     */
    T get_weight(typename std::vector<T>::size_type idx) const
    {
        return m_weights[idx];
    }

    /// Gets the weights
    /**
     * Gets the values of all the weights
     *
     * @return an std::vector containing all the weights
     */
    const std::vector<T> &get_weights() const
    {
        return m_weights;
    }

    /// Randomises all weights
    /**
     * Set all weights to a normally distributed number
     *
     * @param[in] mean the mean of the normal distribution
     * @param[in] std the standard deviation of the normal distribution
     * @param[in] seed the seed to generate the new weights (by default its randomly generated)
     *
     */
    void randomise_weights(double mean = 0., double std = 0.1,
                           decltype(std::random_device{}()) seed = std::random_device{}())
    {
        std::mt19937 gen{seed};
        std::normal_distribution<T> nd{mean, std};
        for (auto &w : m_weights) {
            w = nd(gen);
        }
    }

    /// Sets a bias
    /**
     * Sets a node bias to a new value
     *
     * @param[in] idx index of the bias to be changed.
     *
     */
    void set_bias(typename std::vector<T>::size_type idx, const T &w)
    {
        m_biases[idx] = w;
    }

    /// Sets all biases
    /**
     * Sets all the nodes biases at once
     *
     * @param[in] bs an std::vector containing all the biases to set
     *
     * @throws std::invalid_argument if the input vector dimension is not valid (r*c)
     */
    void set_biases(const std::vector<T> &bs)
    {
        if (bs.size() != m_biases.size()) {
            throw std::invalid_argument("The vector of biases has the wrong dimension");
        }
        m_biases = bs;
    }

    /// Gets a bias
    /**
     * Gets the value of a bias
     *
     * @param[in] idx index of the bias
     *
     */
    T get_bias(typename std::vector<T>::size_type idx) const
    {
        return m_biases[idx];
    }

    /// Gets the biases
    /**
     * Gets the values of all the biases
     *
     * @return an std::vector containing all the biases
     */
    const std::vector<T> &get_biases() const
    {
        return m_biases;
    }

    /// Randomises all biases
    /**
     * Set all biases to a normally distributed number
     *
     * @param[in] mean the mean of the normal distribution
     * @param[in] std the standard deviation of the normal distribution
     * @param[in] seed the seed to generate the new biases (by default its randomly generated)
     *
     */
    void randomise_biases(double mean = 0., double std = 0.1,
                          decltype(std::random_device{}()) seed = std::random_device{}())
    {
        std::mt19937 gen{seed};
        std::normal_distribution<T> nd{mean, std};
        for (auto &b : m_biases) {
            b = nd(gen);
        }
    }

    /*@}*/

protected:
    // For numeric computations
    template <typename U, typename std::enable_if<std::is_same<U, double>::value || is_gdual<U>::value, int>::type = 0>
    U kernel_call(std::vector<U> &function_in, unsigned idx, unsigned weight_idx, unsigned bias_idx) const
    {
        // Weights (we transform the inputs a,b,c,d,e in w_1 a, w_2 b, w_3 c, etc...)
        for (auto j = 0u; j < this->get_arity(); ++j) {
            function_in[j] = function_in[j] * m_weights[weight_idx + j];
        }
        // Biases (we add to the first input a bias so that a,b,c,d,e goes in c, etc...))
        function_in[0] += m_biases[bias_idx];
        // We compute the node function that will, for example, map w_1 a + bias, w_2 b, w_3 c,... into f(w_1 a + w_2 b
        // + w_3 c + ... + bias)
        return this->get_f()[this->get()[idx]](function_in);
    }

    // For the symbolic expression
    template <typename U, typename std::enable_if<std::is_same<U, std::string>::value, int>::type = 0>
    U kernel_call(std::vector<U> &function_in, unsigned idx, unsigned weight_idx, unsigned bias_idx) const
    {
        // Weights
        for (auto j = 0u; j < this->get_arity(); ++j) {
            function_in[j] = "(" + m_weights_symbols[weight_idx + j] + "*" + function_in[j] + ")";
        }
        // Biases
        function_in[0] = "(" + m_biases_symbols[bias_idx] + " + " + function_in[0] + ")";
        return this->get_f()[this->get()[idx]](function_in);
    }

    // overload for the operator () no derivatives
    template <typename U, functor_enabler<U> = 0>
    std::unordered_map<unsigned, U> fill_nodes(const std::vector<U> &in) const
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("Input size is incompatible");
        }
        std::unordered_map<unsigned, U> node;
        std::vector<U> function_in(this->get_arity());
        for (auto i : this->get_active_nodes()) {
            if (i < this->get_n()) {
                node[i] = in[i];
            } else {
                // position in the chromosome of the current node
                unsigned idx = (i - this->get_n()) * (this->get_arity() + 1);
                // starting position in m_weights of the weights relative to the node
                unsigned weight_idx = (i - this->get_n()) * this->get_arity();
                // starting position in m_biases of the node bias
                unsigned bias_idx = i - this->get_n();
                for (auto j = 0u; j < this->get_arity(); ++j) {
                    function_in[j] = node[this->get()[idx + j + 1]];
                }
                node[i] = kernel_call(function_in, idx, weight_idx, bias_idx);
            }
        }
        return node;
    }

    // Overload for the backprop (derivatives)
    template <typename U, functor_enabler<U> = 0>
    void fill_nodes(const std::vector<U> &in, std::unordered_map<unsigned, U> &node,
                    std::unordered_map<unsigned, U> &d_node) const
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("Input size is incompatible");
        }
        std::vector<U> function_in(this->get_arity());
        for (auto i : this->get_active_nodes()) {
            if (i < this->get_n()) {
                node[i] = in[i];
            } else {
                // position in the chromosome of the current node
                unsigned idx = (i - this->get_n()) * (this->get_arity() + 1);
                // starting position in m_weights of the weights relative to the node
                unsigned weight_idx = (i - this->get_n()) * this->get_arity();
                // starting position in m_biases of the node bias
                unsigned bias_idx = i - this->get_n();
                for (auto j = 0u; j < this->get_arity(); ++j) {
                    function_in[j] = node[this->get()[idx + j + 1]];
                }
                node[i] = kernel_call(function_in, idx, weight_idx, bias_idx);
                // sigmoid derivative is sig(1-sig)
                if (this->get_f()[this->get()[idx]].get_name() == "sig") {
                    d_node[i] = node[i] * (1 - node[i]);
                    // tanh derivative is 1 - tanh**2
                } else if (this->get_f()[this->get()[idx]].get_name() == "tanh") {
                    d_node[i] = (1 - node[i] * node[i]);
                    // Relu derivative is 0 if relu<0, 1 otherwise
                } else if (this->get_f()[this->get()[idx]].get_name() == "ReLu") {
                    d_node[i] = (node[i] > 0) ? 1. : 0.;
                }
            }
        }
        return;
    }

    // This overrides the base class update_data_structures and update also the m_connected (as well as m_active_nodes and genes)
    void update_data_structures()
    {
        expression<T>::update_data_structures();
        m_connected.clear();
        for (auto node_id : this->get_active_nodes()) {
            // position in the chromosome of the current node
            unsigned idx = (node_id - this->get_n()) * (this->get_arity() + 1);
            for (auto i = idx + 1; i < idx + 1 + this->get_arity(); ++i) {
                if (this->is_active(this->get()[i])) {
                    m_connected[this->get()[i]].push_back(
                        {node_id, (node_id - this->get_n()) * this->get_arity() + i - idx - 1});
                }
            }
        }
    }

private:
    std::vector<T> m_weights;
    std::vector<std::string> m_weights_symbols;

    std::vector<T> m_biases;
    std::vector<std::string> m_biases_symbols;

    std::unordered_map<unsigned, std::vector<std::pair<unsigned, unsigned>>> m_connected;
}; // namespace dcgp

} // end of namespace dcgp

#endif // DCGP_EXPRESSION_H
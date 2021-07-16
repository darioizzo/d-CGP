#ifndef DCGP_EXPRESSION_ANN_H
#define DCGP_EXPRESSION_ANN_H

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include <audi/io.hpp>

#include <dcgp/config.hpp>
#include <dcgp/expression.hpp>
#include <dcgp/kernel.hpp>
#include <dcgp/s11n.hpp>
#include <dcgp/type_traits.hpp>

namespace dcgp
{

/// A dCGP-ANN expression
/**
 * This class represents an artificial neural network as a differentiable Cartesian Genetic
 * program. It adds weights, biases and backward automated differentiation to the class
 * dcgp::expression.
 *
 */
class expression_ann : public expression<double>
{

private:
    template <typename U>
    using enable_double_string =
        typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::string>::value, int>::type;
    template <typename U>
    using enable_double = typename std::enable_if<std::is_same<U, double>::value, int>::type;

public:
    /// Allowed kernels (for backpropagation to work)
    enum class kernel_type {
        /// sigmoid
        SIG,
        /// Hyperbolic tangent
        TANH,
        /// Rectified linear unit
        RELU,
        /// Exponential linear unit
        ELU,
        /// ISRU
        ISRU,
        /// Simple sum of inputs
        SUM,
        /// non unary sine
        SIN_NU,
        /// non unary cosine
        COS_NU,
        /// non unary cosine
        GAUSSIAN_NU,
        /// negative of the input sum
        INV_SUM,
        /// absolute value of inputs
        ABS,
        /// step funxtion
        STEP
    };
    /// Constructor
    /** Constructs a dCGPANN expression
     *
     * @param[in] n number of inputs (independent variables).
     * @param[in] m number of outputs (dependent variables).
     * @param[in] r number of rows of the cartesian representation of the network as an acyclic graph.
     * @param[in] c number of columns of the cartesian representation of the network as an acyclic graph.
     * @param[in] l number of levels-back allowed. This, essentially, controls the minimum number of allowed
     *  operations in the network. If uncertain set it to c + 1
     * @param[in] arity arities of the basis functions for each column.
     * @param[in] f function set. An std::vector of dcgp::kernel<expression::type>. Can only contain allowed functions.
     * @param[in] seed seed for the random number generator (initial expression and mutations depend on this).
     */
    expression_ann(unsigned n,                    // n. inputs
                   unsigned m,                    // n. outputs
                   unsigned r,                    // n. rows
                   unsigned c,                    // n. columns
                   unsigned l,                    // n. levels-back
                   std::vector<unsigned> arity,   // basis functions' arity
                   std::vector<kernel<double>> f, // functions
                   unsigned seed = dcgp::random_device::next())
        : expression<double>(n, m, r, c, l, arity, f, 0u, seed), m_biases(r * c, 0.), m_kernel_map(f.size())

    {
        // Sanity checks
        for (const auto &ker : f) {
            if (ker.get_name() != "tanh" && ker.get_name() != "sig" && ker.get_name() != "ReLu"
                && ker.get_name() != "ELU" && ker.get_name() != "ISRU" && ker.get_name() != "sin_nu"
                && ker.get_name() != "cos_nu" && ker.get_name() != "gaussian_nu" && ker.get_name() != "sum"
                && ker.get_name() != "inv_sum" && ker.get_name() != "abs" && ker.get_name() != "step") {
                throw std::invalid_argument("Only tanh, sig, ReLu, ELU, ISRU, sin_nu, cos_nu, gaussian_nu, abs, step "
                                            "and sum, inv_sum Kernels are valid "
                                            "for dCGP-ANN expressions");
            }
        }
        // Initialize the kernel map
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            if (f[i].get_name() == "sig") {
                m_kernel_map[i] = kernel_type::SIG;
            } else if (f[i].get_name() == "tanh") {
                m_kernel_map[i] = kernel_type::TANH;
            } else if (f[i].get_name() == "ReLu") {
                m_kernel_map[i] = kernel_type::RELU;
            } else if (f[i].get_name() == "ELU") {
                m_kernel_map[i] = kernel_type::ELU;
            } else if (f[i].get_name() == "ISRU") {
                m_kernel_map[i] = kernel_type::ISRU;
            } else if (f[i].get_name() == "sum") {
                m_kernel_map[i] = kernel_type::SUM;
            } else if (f[i].get_name() == "sin_nu") {
                m_kernel_map[i] = kernel_type::SIN_NU;
            } else if (f[i].get_name() == "cos_nu") {
                m_kernel_map[i] = kernel_type::COS_NU;
            } else if (f[i].get_name() == "gaussian_nu") {
                m_kernel_map[i] = kernel_type::GAUSSIAN_NU;
            } else if (f[i].get_name() == "inv_sum") {
                m_kernel_map[i] = kernel_type::INV_SUM;
            } else if (f[i].get_name() == "abs") {
                m_kernel_map[i] = kernel_type::ABS;
            } else if (f[i].get_name() == "step") {
                m_kernel_map[i] = kernel_type::STEP;
            }
        }
        // Default initialization of weights to 1.
        unsigned n_connections = std::accumulate(this->get_arity().begin(), this->get_arity().end(), 0u) * r;
        m_weights = std::vector<double>(n_connections, 1.);

        // Filling in the symbols for the weights and biases
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            for (auto j = 0u; j < this->_get_arity(node_id); ++j) {
                m_weights_symbols.push_back("w" + std::to_string(node_id) + "_" + std::to_string(j));
            }
        }
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            m_biases_symbols.push_back("b" + std::to_string(node_id));
        }
        // This will call the derived class method (not the base class) where the base class method is also called.
        // As a consequence data members of both classes will be updated.
        update_data_structures();
    }

    /// Constructor
    /** Constructs a dCGPANN expression
     *
     * @param[in] n number of inputs (independent variables).
     * @param[in] m number of outputs (dependent variables).
     * @param[in] r number of rows of the cartesian representation of the network as an acyclic graph.
     * @param[in] c number of columns of the cartesian representation of the network as an acyclic graph.
     * @param[in] l number of levels-back allowed. This, essentially, controls the minimum number of allowed
     *  operations in the network. If uncertain set it to c + 1
     * @param[in] arity uniform arity for all basis functions.
     * @param[in] f function set. An std::vector of dcgp::kernel<expression::type>. Can only contain allowed functions.
     * @param[in] seed seed for the random number generator (initial expression and mutations depend on this).
     */
    expression_ann(unsigned n = 1u,                                               // n. inputs
                   unsigned m = 1u,                                               // n. outputs
                   unsigned r = 1u,                                               // n. rows
                   unsigned c = 1u,                                               // n. columns
                   unsigned l = 1u,                                               // n. levels-back
                   unsigned arity = 2u,                                           // basis functions' arity
                   std::vector<kernel<double>> f = kernel_set<double>({"sum"})(), // functions
                   unsigned seed = dcgp::random_device::next())
        : expression<double>(n, m, r, c, l, std::vector<unsigned>(c, arity), f, 0u, seed), m_biases(r * c, 0.),
          m_kernel_map(f.size())

    {
        // Sanity checks
        for (const auto &ker : f) {
            if (ker.get_name() != "tanh" && ker.get_name() != "sig" && ker.get_name() != "ReLu"
                && ker.get_name() != "ELU" && ker.get_name() != "ISRU" && ker.get_name() != "sin_nu"
                && ker.get_name() != "cos_nu" && ker.get_name() != "gaussian_nu" && ker.get_name() != "sum"
                && ker.get_name() != "inv_sum" && ker.get_name() != "abs" && ker.get_name() != "step") {
                throw std::invalid_argument("Only tanh, sig, ReLu, ELU, ISRU, sin_nu, cos_nu, gaussian_nu, abs, step "
                                            "and sum, inv_sum Kernels are valid "
                                            "for dCGP-ANN expressions");
            }
        }
        // Initialize the kernel map
        for (decltype(f.size()) i = 0u; i < f.size(); ++i) {
            if (f[i].get_name() == "sig") {
                m_kernel_map[i] = kernel_type::SIG;
            } else if (f[i].get_name() == "tanh") {
                m_kernel_map[i] = kernel_type::TANH;
            } else if (f[i].get_name() == "ReLu") {
                m_kernel_map[i] = kernel_type::RELU;
            } else if (f[i].get_name() == "ELU") {
                m_kernel_map[i] = kernel_type::ELU;
            } else if (f[i].get_name() == "ISRU") {
                m_kernel_map[i] = kernel_type::ISRU;
            } else if (f[i].get_name() == "sum") {
                m_kernel_map[i] = kernel_type::SUM;
            } else if (f[i].get_name() == "sin_nu") {
                m_kernel_map[i] = kernel_type::SIN_NU;
            } else if (f[i].get_name() == "cos_nu") {
                m_kernel_map[i] = kernel_type::COS_NU;
            } else if (f[i].get_name() == "gaussian_nu") {
                m_kernel_map[i] = kernel_type::GAUSSIAN_NU;
            } else if (f[i].get_name() == "inv_sum") {
                m_kernel_map[i] = kernel_type::INV_SUM;
            } else if (f[i].get_name() == "abs") {
                m_kernel_map[i] = kernel_type::ABS;
            } else if (f[i].get_name() == "step") {
                m_kernel_map[i] = kernel_type::STEP;
            }
        }
        // Default initialization of weights to 1.
        unsigned n_connections = std::accumulate(this->get_arity().begin(), this->get_arity().end(), 0u) * r;
        m_weights = std::vector<double>(n_connections, 1.);

        // Filling in the symbols for the weights and biases
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            for (auto j = 0u; j < this->_get_arity(node_id); ++j) {
                m_weights_symbols.push_back("w" + std::to_string(node_id) + "_" + std::to_string(j));
            }
        }
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            m_biases_symbols.push_back("b" + std::to_string(node_id));
        }
        // This will call the derived class method (not the base class) where the base class method is also called.
        // As a consequence data members of both classes will be updated.
        update_data_structures();
    }

    /// Evaluates the dCGP-ANN expression
    /**
     * This evaluates the dCGP-ANN expression. This method overrides the base class
     * method. NOTE we cannot template this and the following function as they are virtual in the base class.
     *
     * @param[point] in an std::vector containing the values where the dCGP-ANN expression has
     * to be computed
     *
     * @return The value of the output (an std::vector)
     */
    std::vector<double> operator()(const std::vector<double> &point) const override
    {
        std::vector<double> retval(this->get_m());
        auto node = fill_nodes(point);
        for (auto i = 0u; i < this->get_m(); ++i) {
            retval[i] = node[this->get()[this->get().size() - this->get_m() + i]];
        }
        return retval;
    }
    /// Evaluates the dCGP-ANN expression
    /**
     * This evaluates the dCGP-ANN expression. This method overrides the base class
     * method.
     *
     * @param[point] in an std::vector containing the symbol names.
     *
     * @return The symbolic value of the output (an std::vector)
     */
    std::vector<std::string> operator()(const std::vector<std::string> &point) const override
    {
        std::vector<std::string> retval(this->get_m());
        auto node = fill_nodes(point);
        for (auto i = 0u; i < this->get_m(); ++i) {
            retval[i] = node[this->get()[this->get().size() - this->get_m() + i]];
        }
        return retval;
    }

    /// Evaluates the dCGP-ANN expression
    /**
     * This evaluates the dCGP-ANN expression. This template can be instantiated
     * with type *U* double, in which case the algorithm computes the numerical value of the inputs
     * or with *U* being a string, in which case the instantiated method will produce a symbolic representation of the
     * output.
     *
     * @param[point] in an initialzer list containing the values where the dCGP-ANN expression has
     * to be computed (doubles or strings)
     *
     * @return The value of the output (an std::vector)
     */
    template <typename U, enable_double_string<U> = 0>
    std::vector<U> operator()(const std::initializer_list<U> &point) const
    {
        std::vector<U> dummy(point);
        return (*this)(dummy);
    }

    /// Cumulates the loss and its gradient (of a single point)
    /**
     * Cumulates the loss and its gradient with respect to weights and biases. The values are cumulated into the inputs.
     * If called in a loop with many data points will cumulate the total batch values.
     *
     * @param[value] The initial loss
     * @param[gweights] The initial loss gradient w.r.t. weights
     * @param[gbiases] The initial loss gradient w.r.t. biases
     * @param[point] The input data (single point)
     * @param[prediction] The predicted output (single point)
     * @param[loss_e] The loss type. Must be loss_type::MSE for Mean Square Error (regression) or loss_type::CE for
     * Cross Entropy (classification)
     */
    void d_loss(double &value, std::vector<double> &gweights, std::vector<double> &gbiases,
                const std::vector<double> &point, const std::vector<double> &prediction,
                const expression<double>::loss_type loss_e) const
    {
        if (point.size() != this->get_n()) {
            throw std::invalid_argument("When computing the loss the point dimension (input) seemed wrong, it was: "
                                        + std::to_string(point.size())
                                        + " while I expected: " + std::to_string(this->get_n()));
        }
        if (prediction.size() != this->get_m()) {
            throw std::invalid_argument(
                "When computing the loss the prediction dimension (output) seemed wrong, it was: "
                + std::to_string(prediction.size()) + " while I expected: " + std::to_string(this->get_m()));
        }
        if (gweights.size() != m_weights.size()) {
            throw std::invalid_argument("The size of the return value gweights is: " + std::to_string(gweights.size())
                                        + " while I expected: " + std::to_string(m_weights.size()));
        }

        if (gbiases.size() != m_biases.size()) {
            throw std::invalid_argument("The size of the return value gweights is: " + std::to_string(gbiases.size())
                                        + " while I expected: " + std::to_string(m_biases.size()));
        }

        // ------------------------------------------ Forward pass (takes roughly half of the time) --------------------
        // All active nodes outputs get computed as well as
        // the activation function derivatives
        auto n_nodes = this->get_n() + this->get_r() * this->get_c();
        std::vector<double> node(n_nodes, 0.), d_node(n_nodes, 0.);
        fill_nodes(point, node, d_node); // here is where the computatinal graph is computed.

        // We add to node_d some virtual nodes containing the derivative of the loss with respect to the outputs
        // (dL/do_i)
        switch (loss_e) {
            // Mean Square Error
            case expression<double>::loss_type::MSE: {
                auto sample_dim = static_cast<double>(prediction.size());
                for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
                    auto node_idx = this->get()[this->get().size() - this->get_m() + i];
                    auto dummy = (node[node_idx] - prediction[i]);
                    d_node.push_back(2. * dummy / sample_dim);
                    value += dummy * dummy / sample_dim;
                }
                break; // and exits the switch
            }
            // Cross Entropy
            case expression<double>::loss_type::CE: {
                std::vector<double> ps(this->get_m(), 0.);
                // We store output values in ps
                for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
                    auto node_idx = this->get()[this->get().size() - this->get_m() + i];
                    ps[i] = node[node_idx];
                }
                // We guard from numerical instabilities subtracting the max
                auto max = *std::max_element(ps.begin(), ps.end());
                std::transform(ps.begin(), ps.end(), ps.begin(), [max](double a) { return std::exp(a - max); });
                // We compute the sum of exp(o_i - max)
                double cumsum = std::accumulate(ps.begin(), ps.end(), 0.);
                // We transform to probabilities p_i
                std::transform(ps.begin(), ps.end(), ps.begin(), [cumsum](double a) { return a / cumsum; });
                // We add the derivatives of the loss w.r.t. to outputs
                for (decltype(ps.size()) i = 0u; i < ps.size(); ++i) {
                    d_node.push_back(ps[i] - prediction[i]);
                }
                // We compute the cross-entropy
                std::transform(ps.begin(), ps.end(), prediction.begin(), ps.begin(),
                               [](double p, double y) { return std::log(p) * y; });
                // - sum log(p_i) y_i
                value += -std::accumulate(ps.begin(), ps.end(), 0.);
                break;
            }
        }

        // ------------------------------------------ Backward pass (takes roughly the remaining half)
        // ----------------- We iterate backward on all the active nodes (except the input nodes) filling up the
        // gradient information at each node for the incoming weights and relative bias
        for (auto it = this->get_active_nodes().rbegin(); it != this->get_active_nodes().rend(); ++it) {
            if (*it < this->get_n()) continue;
            // index in the node/d_node vectors
            auto node_id = *it;
            // index of the node in the bias vector
            auto b_idx = node_id - this->get_n();
            // index of the node in the chromosome
            auto c_idx = this->get_gene_idx()[node_id];
            // index of the node in the weight vector
            auto w_idx = c_idx - (node_id - this->get_n());

            // We update the d_node information
            double cum = 0.;
            for (auto i = 0u; i < m_connected[node_id].size(); ++i) {
                // If the node is not "virtual", that is not one of the m virtual nodes we added computing (x-x_i)^2
                if (m_connected[node_id][i].first < this->get_n() + this->get_r() * this->get_c()) {
                    cum += m_weights[m_connected[node_id][i].second] * d_node[m_connected[node_id][i].first];
                } else {
                    auto n_out = m_connected[node_id][i].first - (this->get_n() + this->get_r() * this->get_c());
                    cum += d_node[d_node.size() - this->get_m() + n_out];
                }
            }
            d_node[node_id] *= cum;

            // fill gradients for weights and biases info
            for (auto i = 0u; i < this->_get_arity(node_id); ++i) {
                gweights[w_idx + i] += d_node[node_id] * node[this->get()[c_idx + 1 + i]];
            }
            gbiases[b_idx] += d_node[node_id];
        }
    }

    /// Evaluates the loss and its gradient  (on a batch)
    /**
     * Returns the loss and its gradient with respect to weights and biases.
     *
     * @param[points] The input data (a batch).
     * @param[labels] The predicted outputs (a batch).
     * @param[loss_e] The loss type. Must be loss_type::MSE for Mean Square Error (regression) or loss_type::CE for
     * Cross Entropy (classification)
     * @param[parallel] sets the grain for parallelism. 0 -> no parallelism n -> divides the data into n parts and
     * processes them in parallel threads.
     * @return the loss, the gradient of the loss w.r.t. all weights (also inactive) and the gradient of the loss w.r.t
     * all biases.
     */
    std::tuple<double, std::vector<double>, std::vector<double>> d_loss(const std::vector<std::vector<double>> &points,
                                                                        const std::vector<std::vector<double>> &labels,
                                                                        expression<double>::loss_type loss_e,
                                                                        unsigned parallel = 0u)
    {
        if (points.size() != labels.size()) {
            throw std::invalid_argument("Data and label size mismatch data size is: " + std::to_string(points.size())
                                        + " while label size is: " + std::to_string(labels.size()));
        }
        if (points.size() == 0) {
            throw std::invalid_argument("Data size cannot be zero");
        }
        return d_loss(points.begin(), points.end(), labels.begin(), loss_e, parallel);
    }

    /// Stochastic gradient descent
    /**
     * Performs one "epoch" of stochastic gradient descent using mean square error
     *
     * @param[points] The input data (a batch). Will be randomly shuffled (with labels) after a call to sgd.
     * @param[labels] The predicted outputs (a batch). Will be randomly shuffled (with points) after a call to sgd.
     * @param[lr] The learning rate.
     * @param[batch_size] The batch size.
     * @param[loss_s] A string defining the loss type. Can be one of "MSE" (mean squared error) or "CE" (cross-entropy)
     * @param[parallel] sets the grain for parallelism. 0 -> no parallelism n -> divides the data into n parts and
     * processes them in parallel threads.
     * @param[shuffle] when true it shuffles the points and labels before performing one epoch of training.
     *
     * @return The average error across the batches. Note: this will not be equal to the error on the whole data set
     * as weights get updated after each batch. It is an indicator, though, and its free to compute.
     *
     * @throws std::invalid_argument if the *data* and *label* size do not match or is zero, or if *lr* is not
     * positive.
     */
    double sgd(std::vector<std::vector<double>> &points, std::vector<std::vector<double>> &labels, double lr,
               unsigned batch_size, const std::string &loss_s, unsigned parallel = 0u, bool shuffle = true)
    {
        // Sanity checks for the inputs
        if (points.size() != labels.size()) {
            throw std::invalid_argument("Data and label size mismatch data size is: " + std::to_string(points.size())
                                        + " while label size is: " + std::to_string(labels.size()));
        }
        if (points.size() == 0) {
            throw std::invalid_argument("Data size cannot be zero");
        }
        if (lr <= 0) {
            throw std::invalid_argument("The learning rate must be a positive number, while: " + std::to_string(lr)
                                        + " was detected.");
        }

        // Decoding the loss from string to the enum type (loss_s -> loss_e)
        expression<double>::loss_type loss_e;
        if (loss_s == "MSE") {
            loss_e = expression<double>::loss_type::MSE;
        } else if (loss_s == "CE") {
            loss_e = expression<double>::loss_type::CE;
        } else {
            throw std::invalid_argument("The requested loss was: " + loss_s + " while only MSE and CE are allowed");
        }

        if (shuffle) {
            // Creating a shuffle
            // Create two random engines with the same state
            auto seed = std::random_device{}();
            std::mt19937 eng1(seed);
            auto eng2 = eng1;
            std::shuffle(points.begin(), points.end(), eng1);
            std::shuffle(labels.begin(), labels.end(), eng2);
        }

        // Starting the iteration
        auto dfirst = points.begin();
        auto dlast = points.end();
        auto lfirst = labels.begin();
        double retval = 0.;
        double counter = 0.;
        while (dfirst != dlast) {
            if (dfirst + batch_size > dlast) {
                retval += update_weights(dfirst, dlast, lfirst, lr, loss_e, parallel);
                dfirst = dlast;
                counter++;
            } else {
                retval += update_weights(dfirst, dfirst + batch_size, lfirst, lr, loss_e, parallel);
                dfirst += batch_size;
                lfirst += batch_size;
                counter++;
            }
        }
        return retval / counter;
    }

    /// Sets the output nonlinearities
    /**
     * Sets the nonlinearities of all nodes connected to the output nodes.
     * This is useful when, for example, the dCGPANN is used for a regression task where output values are expected in
     * [-1 1] and hence the output layer should have some sigmoid or tanh nonlinearity.
     *
     *
     * @param[in] name the name of the kernel (nonlinearity)
     *
     * @throw std::invalid_argument if *name* is invalid.
     */
    void set_output_f(const std::string &name)
    {
        typename std::vector<kernel_type>::iterator it;
        if (name == "sig") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::SIG);
        } else if (name == "tanh") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::TANH);
        } else if (name == "ReLu") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::RELU);
        } else if (name == "ELU") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::ELU);
        } else if (name == "ISRU") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::ISRU);
        } else if (name == "sum") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::SUM);
        } else if (name == "sin_nu") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::SIN_NU);
        } else if (name == "cos_nu") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::COS_NU);
        } else if (name == "gaussian_nu") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::GAUSSIAN_NU);
        } else if (name == "inv_sum") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::INV_SUM);
        } else if (name == "abs") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::ABS);
        } else if (name == "step") {
            it = std::find(m_kernel_map.begin(), m_kernel_map.end(), kernel_type::STEP);
        }

        if (it == m_kernel_map.end()) {
            throw std::invalid_argument("The nonlinearity " + name + " is not a Kernel for this dCGP expression");
        } else {
            unsigned f_id = static_cast<unsigned>(it - m_kernel_map.begin());
            for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
                this->set_f_gene(this->get()[this->get().size() - 1 - i], f_id);
            }
        }
    }
    /// Computes the number of weights influencing the result
    /**
     * Computes the number of weights influencing the result. This will also be the number
     * of weights that are updated when calling sgd. The number of active weights, as well as
     * the number of active nodes, define the complexity of the expression expressed by the chromosome.
     *
     * @param[in] unique when true weights are counted only once if connecting the same two nodes.
     */
    unsigned n_active_weights(bool unique = false) const
    {
        unsigned retval = 0u;
        for (auto node_id : this->get_active_nodes()) {
            if (node_id < this->get_n()) continue;
            if (!unique) {
                retval += this->_get_arity(node_id);
            } else {
                auto g_idx = this->get_gene_idx()[node_id];
                auto arity = this->_get_arity(node_id);
                std::vector<unsigned> con_id(this->get().begin() + g_idx + 1u,
                                             this->get().begin() + g_idx + 1u + arity);
                std::sort(con_id.begin(), con_id.end());
                retval += static_cast<unsigned>(std::unique(con_id.begin(), con_id.end()) - con_id.begin());
            }
        }
        return retval;
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
        audi::stream(os, "\tNumber of rows:\t\t\t", d.get_r(), '\n');
        audi::stream(os, "\tNumber of columns:\t\t", d.get_c(), '\n');
        audi::stream(os, "\tNumber of levels-back allowed:\t", d.get_l(), '\n');
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
    void set_weight(unsigned node_id, unsigned input_id, const double &w)
    {
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_r() * this->get_c()) {
            throw std::invalid_argument("Requested node id does not exist");
        }
        if (input_id >= this->_get_arity(node_id)) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }
        // index of the node in the weight vector
        auto idx = this->get_gene_idx()[node_id] - (node_id - this->get_n()) + input_id;
        m_weights[idx] = w;
    }

    /// Sets a weight
    /**
     * Sets a connection weight to a new value
     *
     * @param[idx] index of the weight to be changed.
     * @param[w] value of the weight to be changed.
     *
     * @throws std::invalid_argument if the node_id or input_id are not valid
     */
    void set_weight(std::vector<double>::size_type idx, const double &w)
    {
        m_weights[idx] = w;
    }

    /// Sets all weights
    /**
     * Sets all the connection weights at once
     *
     * @param[in] ws an std::vector containing all the weights to set
     *
     * @throws std::invalid_argument if the input vector dimension is not valid.
     */
    void set_weights(const std::vector<double> &ws)
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
     * @param[in] input_id the id of the node input (0 up to node arity-1)
     *
     * @return the value of the weight
     *
     * @throws std::invalid_argument if the node_id or input_id are not valid
     */
    double get_weight(unsigned node_id, unsigned input_id) const
    {
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_r() * this->get_c()) {
            throw std::invalid_argument(
                "Requested node id does not exist or does not have a weight (e.g. input nodes)");
        }
        if (input_id >= this->_get_arity(node_id)) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }

        auto idx = this->get_gene_idx()[node_id] - (node_id - this->get_n()) + input_id;
        return m_weights[idx];
    }

    /// Gets a weight
    /**
     * Gets the value of a connection weight
     *
     * @param[in] idx index of the weight
     *
     */
    double get_weight(std::vector<double>::size_type idx) const
    {
        return m_weights[idx];
    }

    /// Gets the weights
    /**
     * Gets the values of all the weights.
     *
     * @return an std::vector containing all the weights
     */
    const std::vector<double> &get_weights() const
    {
        return m_weights;
    }

/// Randomises all weights
/**
 * Set all weights to a normally distributed number
 *
 * @param[mean] the mean of the normal distribution.
 * @param[std] the standard deviation of the normal distribution.
 * @param[seed] the seed to generate the new weights (by default its randomly generated).
 *
 */
#if !defined(DCGP_DOXYGEN_INVOKED)
    void randomise_weights(double mean = 0, double std = 0.1,
                           std::random_device::result_type seed = std::random_device{}())
    {
        std::mt19937 gen{seed};
        std::normal_distribution<double> nd{mean, std};
        for (auto &w : m_weights) {
            w = nd(gen);
        }
    }
#else
    void randomise_weights(double mean = 0, double std = 0.1, std::random_device::result_type seed = random_number) {}
#endif

    /// Sets a bias
    /**
     * Sets a node bias to a new value
     *
     * @param[idx] index of the bias to be changed.
     * @param[w] value of the new bias.
     *
     */
    void set_bias(typename std::vector<double>::size_type idx, const double &w)
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
    void set_biases(const std::vector<double> &bs)
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
    double get_bias(typename std::vector<double>::size_type idx) const
    {
        return m_biases[idx];
    }

    /// Gets the biases
    /**
     * Gets the values of all the biases
     *
     * @return an std::vector containing all the biases
     */
    const std::vector<double> &get_biases() const
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
#if !defined(DCGP_DOXYGEN_INVOKED)
    void randomise_biases(double mean = 0., double std = 0.1,
                          std::random_device::result_type seed = std::random_device{}())
    {
        std::mt19937 gen{seed};
        std::normal_distribution<double> nd{mean, std};
        for (auto &b : m_biases) {
            b = nd(gen);
        }
    }
#else
    void randomise_biases(double mean = 0, double std = 0.1, std::random_device::result_type seed = random_number) {}
#endif

    /*@}*/

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
        // invoke serialization of the base class
        ar &boost::serialization::base_object<expression<double>>(*this);
        ar &m_weights;
        ar &m_weights_symbols;
        ar &m_biases;
        ar &m_biases_symbols;
        ar &m_connected;
        ar &m_kernel_map;
    }

    // Delete ephemeral constants methods.
    void set_eph_val(const std::vector<double> &) = delete;
    void set_eph_symb(const std::vector<double> &) = delete;

private:
    // For numeric computations
    double kernel_call(std::vector<double> &function_in, unsigned idx, unsigned arity, unsigned weight_idx,
                       unsigned bias_idx) const
    {
        // Weights (we transform the inputs a,b,c,d,e in w_1 a, w_2 b, w_3 c, etc...)
        for (auto j = 0u; j < arity; ++j) {
            function_in[j] = function_in[j] * m_weights[weight_idx + j];
        }
        // Biases (we add to the first input a bias so that a,b,c,d,e goes in c, etc...))
        function_in[0] += m_biases[bias_idx];
        // We compute the node function that will, for example, map w_1 a + bias, w_2 b, w_3 c,... into f(w_1 a +
        // w_2 b + w_3 c + ... + bias)
        return this->get_f()[this->get()[idx]](function_in);
    }

    // For the symbolic expression
    std::string kernel_call(std::vector<std::string> &function_in, unsigned idx, unsigned arity, unsigned weight_idx,
                            unsigned bias_idx) const
    {
        // Weights
        for (auto j = 0u; j < arity; ++j) {
            function_in[j] = m_weights_symbols[weight_idx + j] + "*" + function_in[j];
        }
        // Biases
        function_in[0] = m_biases_symbols[bias_idx] + "+" + function_in[0];
        return this->get_f()[this->get()[idx]](function_in);
    }

    // computes node to evaluate the expression
    template <typename U, enable_double_string<U> = 0>
    std::vector<U> fill_nodes(const std::vector<U> &in) const
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("Input size is incompatible");
        }
        std::vector<U> node(this->get_n() + this->get_r() * this->get_c());
        std::vector<U> function_in;
        for (auto node_id : this->get_active_nodes()) {
            if (node_id < this->get_n()) {
                node[node_id] = in[node_id];
            } else {
                unsigned arity = this->_get_arity(node_id);
                function_in.resize(arity);
                // position in the chromosome of the current node
                unsigned g_idx = this->get_gene_idx()[node_id];
                // starting position in m_weights of the weights relative to the node
                unsigned w_idx = g_idx - (node_id - this->get_n());
                // starting position in m_biases of the node bias
                unsigned b_idx = node_id - this->get_n();
                for (auto j = 0u; j < arity; ++j) {
                    function_in[j] = node[this->get()[g_idx + j + 1]];
                }
                node[node_id] = kernel_call(function_in, g_idx, arity, w_idx, b_idx);
            }
        }
        return node;
    }

    // computes node and node_d to start backprop
    void fill_nodes(const std::vector<double> &in, std::vector<double> &node, std::vector<double> &d_node) const
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("Input size is incompatible");
        }
        // Start
        std::vector<double> function_in;
        for (auto node_id : this->get_active_nodes()) {
            if (node_id < this->get_n()) {
                node[node_id] = in[node_id];
                // We need d_node to have the same structure of node, hence we also
                // put some bogus entries fot the input nodes that actually do not have an activation function
                // hence no need/use/meaning for a derivative
                d_node[node_id] = 0.;
            } else {
                unsigned arity = this->_get_arity(node_id);
                function_in.resize(arity);
                // position in the chromosome of the current node
                unsigned g_idx = this->get_gene_idx()[node_id];
                // starting position in m_weights of the weights relative to the node
                unsigned w_idx = g_idx - (node_id - this->get_n());
                // starting position in m_biases of the node bias
                unsigned b_idx = node_id - this->get_n();
                for (auto j = 0u; j < arity; ++j) {
                    function_in[j] = node[this->get()[g_idx + j + 1]];
                }
                node[node_id] = kernel_call(function_in, g_idx, arity, w_idx, b_idx);
                // take cares of d_node
                // sigmoid derivative is sig(1-sig)
                switch (m_kernel_map[this->get()[g_idx]]) {
                    case kernel_type::SIG:
                        d_node[node_id] = node[node_id] * (1. - node[node_id]);
                        break;
                    case kernel_type::TANH:
                        d_node[node_id] = 1. - node[node_id] * node[node_id];
                        break;
                    case kernel_type::SUM:
                        d_node[node_id] = 1.;
                        break;
                    case kernel_type::RELU:
                        d_node[node_id] = (node[node_id] > 0.) ? 1. : 0.;
                        break;
                    case kernel_type::ELU:
                        d_node[node_id] = (node[node_id] > 0.) ? 1. : node[node_id] + 1.;
                        break;
                    case kernel_type::ISRU: {
                        auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                        d_node[node_id] = node[node_id] * node[node_id] * node[node_id] / cumin / cumin / cumin;
                        break;
                    }
                    case kernel_type::SIN_NU: {
                        auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                        d_node[node_id] = std::cos(cumin);
                        break;
                    }
                    case kernel_type::COS_NU: {
                        auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                        d_node[node_id] = -std::sin(cumin);
                        break;
                    }
                    case kernel_type::GAUSSIAN_NU: {
                        auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                        d_node[node_id] = -2 * cumin * node[node_id];
                        break;
                    }
                    case kernel_type::INV_SUM: {
                        d_node[node_id] = -1.;
                        break;
                    }
                    case kernel_type::ABS: {
                        auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                        d_node[node_id] = cumin < 0. ? -1 : 1;
                        break;
                    }
                    case kernel_type::STEP: {
                        d_node[node_id] = 0.;
                        break;
                    }
                }
            }
        }
    }

    // This overrides the base class update_data_structures and updates also the m_connected (as well as
    // m_active_nodes and genes). It is called upon construction and each time active genes are changed.
    void update_data_structures() override
    {
        expression<double>::update_data_structures();
        m_connected.clear();
        m_connected.resize(this->get_n() + this->get_m() + this->get_r() * this->get_c());
        for (auto node_id : this->get_active_nodes()) {
            if (node_id >= this->get_n()) { // not for input nodes
                // start in the chromosome of the genes expressing the node_id connections
                unsigned idx = this->get_gene_idx()[node_id] + 1u;
                // start in the weight vector of the genes expressing the node_id connections
                unsigned w_idx = (idx - 1u) - (node_id - this->get_n());
                // loop over the genes representing connections
                for (auto i = 0u; i < this->_get_arity(node_id); ++i) {
                    if (this->is_active_node(this->get()[idx + i])) {
                        m_connected[this->get()[idx + i]].push_back({node_id, w_idx + i});
                    }
                }
            }
        }
        // We now add the output nodes with ids starting from n + r * c. In this case the weight is not
        // relevant, hence we use the arbitrary value 0u as index in the weight vector.
        for (auto i = 0u; i < this->get_m(); ++i) {
            auto virtual_idx = this->get_n() + this->get_r() * this->get_c() + i;
            auto node_idx = this->get()[this->get().size() - this->get_m() + i];
            m_connected[node_idx].push_back({virtual_idx, 0u});
        }
    }

    /// Performs one weight/bias update
    /**
     * Updates m_weights and m_biases using gradient descent
     *
     * @param[dfirst] Start range for the data
     * @param[dlast] End range for the data
     * @param[lfirst] Start range for the labels
     * @param[lr] The learning rate
     * @param[loss_e] The loss type
     * @param[parallel] sets the grain for parallelism. 0 -> no parallelism n -> divides the data into n parts and
     * processes them in parallel threads.
     *
     * @return the loss before the weight update
     *
     */
    double update_weights(typename std::vector<std::vector<double>>::const_iterator dfirst,
                          typename std::vector<std::vector<double>>::const_iterator dlast,
                          typename std::vector<std::vector<double>>::const_iterator lfirst, double lr,
                          expression<double>::loss_type loss_e, unsigned parallel = 0u)
    {
        auto err = d_loss(dfirst, dlast, lfirst, loss_e, parallel);

        // We now update the weights with the stochastic gradient descent update rule
        std::transform(m_weights.begin(), m_weights.end(), std::get<1>(err).begin(), m_weights.begin(),
                       [&lr](double a, double b) { return a - lr * b; });
        std::transform(m_biases.begin(), m_biases.end(), std::get<2>(err).begin(), m_biases.begin(),
                       [&lr](double a, double b) { return a - lr * b; });
        return std::get<0>(err);
    }

    std::tuple<double, std::vector<double>, std::vector<double>>
    d_loss(typename std::vector<std::vector<double>>::const_iterator dfirst,
           typename std::vector<std::vector<double>>::const_iterator dlast,
           typename std::vector<std::vector<double>>::const_iterator lfirst, expression<double>::loss_type loss_e,
           unsigned parallel = 0u) const
    {
        // Batch dimension
        const unsigned batch_size = static_cast<unsigned>(dlast - dfirst);
        // These variables need to be read/written by all tasks.
        double value = 0.;
        std::vector<double> gweights(m_weights.size(), 0.);
        std::vector<double> gbiases(m_biases.size(), 0.);

        if (parallel > 0u) {
            if (batch_size % parallel != 0) {
                throw std::invalid_argument("The batch size is: " + std::to_string(batch_size)
                                            + " and cannot be divided into " + std::to_string(parallel) + " parts.");
            }
            unsigned inner_batch_size = batch_size / parallel;
            // The mutex that will protect read write access to value, gweights, gbiases.
            tbb::spin_mutex mutex_weights_updates;
            // This loops over all points, predictions in the mini-batch
            tbb::parallel_for(0u, batch_size, inner_batch_size, [&](unsigned i) {
                double value2 = 0.;
                std::vector<double> gweights2(m_weights.size(), 0.);
                std::vector<double> gbiases2(m_biases.size(), 0.);
                // The loss and its gradient get computed
                for (auto j = 0u; j < inner_batch_size; ++j) {
                    d_loss(value2, gweights2, gbiases2, *(dfirst + i + j), *(lfirst + i + j), loss_e);
                }
                // We acquire the lock on the mutex
                tbb::spin_mutex::scoped_lock lock(mutex_weights_updates);
                // We update the cumulative loss and gradient
                value += value2;
                std::transform(gweights.begin(), gweights.end(), gweights2.begin(), gweights.begin(),
                               [](double a, double b) { return a + b; });
                std::transform(gbiases.begin(), gbiases.end(), gbiases2.begin(), gbiases.begin(),
                               [](double a, double b) { return a + b; });
            });
        } else {
            for (unsigned i = 0u; i < batch_size; ++i) {
                // The loss and its gradient get computed and cumulated in value, gweights, gbiases
                d_loss(value, gweights, gbiases, *(dfirst + i), *(lfirst + i), loss_e);
            }
        }
        std::transform(gweights.begin(), gweights.end(), gweights.begin(),
                       [&batch_size](double a) { return a / batch_size; });
        std::transform(gbiases.begin(), gbiases.end(), gbiases.begin(),
                       [&batch_size](double a) { return a / batch_size; });
        value /= batch_size;
        return std::make_tuple(std::move(value), std::move(gweights), std::move(gbiases));
    }

private:
    std::vector<double> m_weights;
    std::vector<std::string> m_weights_symbols;

    std::vector<double> m_biases;
    std::vector<std::string> m_biases_symbols;

    // In order to be able to perform backpropagation on the dCGPANN program, we need to add
    // to the usual CGP data structures one that contains for each node the list of nodes
    // (and weights) it feeds into. We also need to add some virtual nodes (to keep track of output nodes
    // dependencies) The assigned virtual ids starting from n + r * c
    std::vector<std::vector<std::pair<unsigned, unsigned>>> m_connected;
    // Kernel map (this is here to avoid string comparisons)
    std::vector<kernel_type> m_kernel_map;
};
} // end of namespace dcgp

#endif // DCGP_EXPRESSION_H

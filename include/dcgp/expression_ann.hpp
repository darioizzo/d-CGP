#ifndef DCGP_EXPRESSION_ANN_H
#define DCGP_EXPRESSION_ANN_H

#include <algorithm>
#include <audi/io.hpp>
#include <initializer_list>
#include <iostream>
#include <map>
#include <numeric>
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
 */
template <typename T>
class expression_ann : public expression<T>
{

private:
    template <typename U>
    using enable_double_string =
        typename std::enable_if<std::is_same<U, double>::value || std::is_same<U, std::string>::value, int>::type;
    template <typename U>
    using enable_double = typename std::enable_if<std::is_same<U, double>::value, int>::type;

public:
    // loss types: Mean Squared Error or Cross Entropy
    enum class loss_type { MSE, CE };

    /// Constructor
    /** Constructs a dCGP expression
     *
     * @param[in] n number of inputs (independent variables).
     * @param[in] m number of outputs (dependent variables).
     * @param[in] r number of rows of the dCGP.
     * @param[in] c number of columns of the dCGP.
     * @param[in] l number of levels-back allowed for the dCGP.
     * @param[in] arity arities of the basis functions for each column.
     * @param[in] f function set. An std::vector of dcgp::kernel<expression::type>.
     * @param[in] seed seed for the random number generator (initial expression and mutations depend on this).
     */
    expression_ann(unsigned n,                  // n. inputs
                   unsigned m,                  // n. outputs
                   unsigned r,                  // n. rows
                   unsigned c,                  // n. columns
                   unsigned l,                  // n. levels-back
                   std::vector<unsigned> arity, // basis functions' arity
                   std::vector<kernel<T>> f,    // functions
                   unsigned seed                // seed for the pseudo-random numbers
                   )
        : expression<T>(n, m, r, c, l, arity, f, seed), m_biases(r * c, T(0.))

    {
        // Sanity checks
        for (const auto &ker : f) {
            if (ker.get_name() != "tanh" && ker.get_name() != "sig" && ker.get_name() != "ReLu"
                && ker.get_name() != "ELU" && ker.get_name() != "ISRU") {
                throw std::invalid_argument(
                    "Only tanh, sig, ReLu, ELU and ISRU Kernels are valid for dCGP-ANN expressions");
            }
        }
        // Default initialization of weights to 1.
        unsigned n_connections = std::accumulate(this->get_arity().begin(), this->get_arity().end(), 0u) * r;
        m_weights = std::vector<T>(n_connections, T(1.));

        // Filling in the symbols for the weights and biases
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            for (auto j = 0u; j < this->get_arity(node_id); ++j) {
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
    /** Constructs a dCGP expression
     *
     * @param[in] n number of inputs (independent variables).
     * @param[in] m number of outputs (dependent variables).
     * @param[in] r number of rows of the dCGP.
     * @param[in] c number of columns of the dCGP.
     * @param[in] l number of levels-back allowed for the dCGP.
     * @param[in] arity uniform arity for all basis functions.
     * @param[in] f function set. An std::vector of dcgp::kernel<expression::type>.
     * @param[in] seed seed for the random number generator (initial expression and mutations depend on this).
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
        : expression<T>(n, m, r, c, l, std::vector<unsigned>(c, arity), f, seed), m_biases(r * c, T(0.))

    {
        // Sanity checks
        for (const auto &ker : f) {
            if (ker.get_name() != "tanh" && ker.get_name() != "sig" && ker.get_name() != "ReLu"
                && ker.get_name() != "ELU" && ker.get_name() != "ISRU") {
                throw std::invalid_argument(
                    "Only tanh, sig, ReLu, ELU and ISRU Kernels are valid for dCGP-ANN expressions");
            }
        }
        // Default initialization of weights to 1.
        unsigned n_connections = std::accumulate(this->get_arity().begin(), this->get_arity().end(), 0u) * r;
        m_weights = std::vector<T>(n_connections, T(1.));

        // Filling in the symbols for the weights and biases
        for (auto node_id = n; node_id < r * c + n; ++node_id) {
            for (auto j = 0u; j < this->get_arity(node_id); ++j) {
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
     * This evaluates the dCGP-ANN expression. According to the template parameter
     * it will compute the value (T) or a symbolic
     * representation (std::string). Any other type will result in a compilation-time
     * error.
     *
     * @param[point] in an std::vector containing the values where the dCGP-ANN expression has
     * to be computed (doubles or strings)
     *
     * @return The value of the output (an std::vector)
     */
    template <typename U, enable_double_string<U> = 0>
    std::vector<U> operator()(const std::vector<U> &point) const
    {
        std::vector<U> retval(this->get_m());
        auto node = fill_nodes(point);
        for (auto i = 0u; i < this->get_m(); ++i) {
            retval[i] = node[this->get()[this->get().size() - this->get_m() + i]];
        }
        return retval;
    }

    /// Evaluates the dCGP-ANN expression
    /**
     * This evaluates the dCGP-ANN expression. According to the template parameter
     * it will compute the value (T) or a symbolic
     * representation (std::string). Any other type will result in a compilation-time
     * error.
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

    /// Evaluates the loss (single data point)
    /**
     * Returns the loss over a single point of data of the dCGPANN output.
     *
     * @param[point] The input data (single point)
     * @param[prediction] The predicted output (single point)
     * @param[loss_e] The loss type. Can be "MSE" for Mean Square Error (regression) or "CE" for Cross Entropy
     * (classification)
     * @return the mse
     */
    double loss(const std::vector<double> &point, const std::vector<double> &prediction, loss_type loss_e)
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
        double retval(0.);

        auto outputs = this->operator()(point);
        switch (loss_e) {
            // Mean Square Error
            case loss_type::MSE:
                for (decltype(outputs.size()) i = 0u; i < outputs.size(); ++i) {
                    retval += (outputs[i] - prediction[i]) * (outputs[i] - prediction[i]);
                }
                break; // and exits the switch
            // Cross Entropy
            case loss_type::CE:
                // exp(a_i)
                std::transform(outputs.begin(), outputs.end(), outputs.begin(), [](double a) { return std::exp(a); });
                // sum exp(a_i)
                double cumsum = std::accumulate(outputs.begin(), outputs.end(), 0.);
                // log(p_i) * y_i
                std::transform(outputs.begin(), outputs.end(), prediction.begin(), outputs.begin(),
                               [cumsum](double a, double y) { return std::log(a / cumsum) * y; });
                // - sum log(p_i) y_i
                retval = -std::accumulate(outputs.begin(), outputs.end(), 0.);
                break;
        }

        return retval;
    }

    /// Evaluates the model loss (on a batch)
    /**
     * Evaluates the model loss over a batch.
     *
     * @param[points] The input data (a batch).
     * @param[labels] The predicted outputs (a batch).
     * @param[loss_e] The loss type. Can be "MSE" for Mean Square Error (regression) or "CE" for Cross Entropy
     * (classification)
     * @return the loss
     */
    double loss(const std::vector<std::vector<double>> &points, const std::vector<std::vector<double>> &labels,
                std::string loss_s)
    {
        if (points.size() != labels.size()) {
            throw std::invalid_argument("Data and label size mismatch data size is: " + std::to_string(points.size())
                                        + " while label size is: " + std::to_string(labels.size()));
        }
        if (points.size() == 0) {
            throw std::invalid_argument("Data size cannot be zero");
        }
        loss_type loss_e;
        if (loss_s == "MSE") { // Mean Square Error
            loss_e = loss_type::MSE;
        } else if (loss_s == "CE") {
            loss_e = loss_type::CE; // Cross Entropy
        } else {
            throw std::invalid_argument("The requested loss was: " + loss_s + " while only MSE and CE are allowed");
        }
        return loss(points.begin(), points.end(), labels.begin(), loss_e);
    }

    /// Evaluates the loss and its gradient (on a single point)
    /**
     * Returns the loss and its gradient with respect to weights and biases.
     *
     * @param[point] The input data (single point)
     * @param[prediction] The predicted output (single point)
     * @param[loss_e] The loss type. Must be loss_type::MSE for Mean Square Error (regression) or loss_type::CE for
     * Cross Entropy (classification)
     * @return the loss, the gradient of the loss w.r.t. all weights (also inactive) and the gradient of the loss w.r.t
     * all biases
     */
    template <typename U, enable_double<U> = 0>
    std::tuple<U, std::vector<U>, std::vector<U>> d_loss(const std::vector<U> &point, const std::vector<U> &prediction,
                                                         const loss_type loss_e)
    {
        if (point.size() != this->get_n()) {
            throw std::invalid_argument("When computing the mse the point dimension (input) seemed wrong, it was: "
                                        + std::to_string(point.size())
                                        + " while I expected: " + std::to_string(this->get_n()));
        }
        if (prediction.size() != this->get_m()) {
            throw std::invalid_argument(
                "When computing the mse the prediction dimension (output) seemed wrong, it was: "
                + std::to_string(prediction.size()) + " while I expected: " + std::to_string(this->get_m()));
        }
        U value(U(0.));
        std::vector<U> gweights(m_weights.size(), U(0.));
        std::vector<U> gbiases(m_biases.size(), U(0.));

        // ------------------------------------------ Forward pass (takes roughly half of the time) --------------------
        // All active nodes outputs get computed as well as
        // the activation function derivatives
        auto n_nodes = this->get_n() + this->get_r() * this->get_c();
        std::vector<U> node(n_nodes, 0.), d_node(n_nodes, 0.);
        fill_nodes(point, node, d_node); // here is where the computations happen.

        // We add to node_d some virtual nodes containing the derivative of the loss with respect to the outputs
        // (dL/do_i)
        switch (loss_e) {
            // Mean Square Error
            case loss_type::MSE:
                for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
                    auto node_idx = this->get()[this->get().size() - this->get_m() + i];
                    auto dummy = (node[node_idx] - prediction[i]);
                    d_node.push_back(2 * dummy);
                    value += dummy * dummy;
                }
                break; // and exits the switch
            // Cross Entropy
            case loss_type::CE:
                std::vector<double> ps(this->get_m(), 0.);
                double cumsum = 0.;
                // We compute exp(a_i) and sum exp(a_i) for softmax
                for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
                    auto node_idx = this->get()[this->get().size() - this->get_m() + i];
                    ps[i] = std::exp(node[node_idx]);
                    cumsum += ps[i];
                }
                // We compute the probabilities p_i = a_i / sum a_i
                std::transform(ps.begin(), ps.end(), ps.begin(), [cumsum](double a) { return a / cumsum; });
                // We add the derivatives of the loss w.r.t. to outputs
                for (decltype(ps.size()) i = 0u; i < ps.size(); ++i) {
                    d_node.push_back(ps[i] - prediction[i]);
                }
                // We compute the cross-entropy
                std::transform(ps.begin(), ps.end(), prediction.begin(), ps.begin(),
                               [](double p, double y) { return std::log(p) * y; });
                // - sum log(p_i) y_i
                value = -std::accumulate(ps.begin(), ps.end(), 0.);
                break;
        }

        // ------------------------------------------ Backward pass (takes roughly the remaining half)
        // ----------------- We iterate backward on all the active nodes (except the input nodes) filling up the
        // gradient information at each node for the incoming weights and relative bias
        for (auto it = this->get_active_nodes().rbegin(); it != this->get_active_nodes().rend(); ++it) {
            if (*it < this->get_n()) continue;
            // index of the node in the bias vector
            auto b_idx = *it - this->get_n();
            // index of the node in the chromosome
            auto c_idx = this->get_gene_idx()[*it];
            // index of the node in the weight vector
            auto w_idx = c_idx - (*it - this->get_n());

            // index in the node/d_node vectors
            auto node_id = *it;
            // We update the d_node information
            U cum = 0.;
            for (auto i = 0u; i < m_connected[*it].size(); ++i) {
                // If the node is not "virtual", that is not one of the m virtual nodes we added computing (x-x_i)^2
                if (m_connected[*it][i].first < this->get_n() + this->get_r() * this->get_c()) {
                    cum += m_weights[m_connected[*it][i].second] * d_node[m_connected[*it][i].first];
                } else {
                    auto n_out = m_connected[*it][i].first - (this->get_n() + this->get_r() * this->get_c());
                    cum += d_node[d_node.size() - this->get_m() + n_out];
                }
            }
            d_node[node_id] *= cum;

            // fill gradients for weights and biases info
            for (auto i = 0u; i < this->get_arity(node_id); ++i) {
                gweights[w_idx + i] = d_node[node_id] * node[this->get()[c_idx + 1 + i]];
            }
            gbiases[b_idx] = d_node[node_id];
        }

        return std::make_tuple(std::move(value), std::move(gweights), std::move(gbiases));
    }

    /// Evaluates the mean square error and its gradient  (on a batch)
    /**
     * Returns the mean squared error and its gradient with respect to weights and biases.
     *
     * @param[points] The input data (a batch).
     * @param[labels] The predicted outputs (a batch).
     * @param[loss_e] The loss type. Must be loss_type::MSE for Mean Square Error (regression) or loss_type::CE for
     * Cross Entropy (classification)
     * @return the loss, the gradient of the loss w.r.t. all weights (also inactive) and the gradient of the loss w.r.t
     * all biases.
     */
    template <typename U, enable_double<U> = 0>
    std::tuple<U, std::vector<U>, std::vector<U>> d_loss(const std::vector<std::vector<U>> &points,
                                                         const std::vector<std::vector<U>> &labels, loss_type loss_e)
    {
        if (points.size() != labels.size()) {
            throw std::invalid_argument("Data and label size mismatch data size is: " + std::to_string(points.size())
                                        + " while label size is: " + std::to_string(labels.size()));
        }
        if (points.size() == 0) {
            throw std::invalid_argument("Data size cannot be zero");
        }
        return d_loss<U>(points.begin(), points.end(), labels.begin(), loss_e);
    }

    /// Stochastic gradient descent
    /**
     * Performs one "epoch" of stochastic gradient descent using mean square error
     *
     * @param[points] The input data (a batch).
     * @param[labels] The predicted outputs (a batch).
     * @param[l_rate] The learning rate.
     * @param[batch_size] The batch size.
     *
     * @throws std::invalid_argument if the *data* and *label* size do not match or is zero, or if *l_rate* is not
     * positive.
     */
    template <typename U, enable_double<U> = 0>
    void sgd(const std::vector<std::vector<U>> &points, const std::vector<std::vector<U>> &labels, double l_rate,
             unsigned batch_size, std::string loss_s)
    {
        if (points.size() != labels.size()) {
            throw std::invalid_argument("Data and label size mismatch data size is: " + std::to_string(points.size())
                                        + " while label size is: " + std::to_string(labels.size()));
        }
        if (points.size() == 0) {
            throw std::invalid_argument("Data size cannot be zero");
        }
        if (l_rate <= 0) {
            throw std::invalid_argument("The learning rate must be a positive number, while: " + std::to_string(l_rate)
                                        + " was detected.");
        }

        loss_type loss_e;
        if (loss_s == "MSE") {
            loss_e = loss_type::MSE;
        } else if (loss_s == "CE") {
            loss_e = loss_type::CE;
        } else {
            throw std::invalid_argument("The requested loss was: " + loss_s + " while only MSE and CE are allowed");
        }

        auto dfirst = points.begin();
        auto dlast = points.end();
        auto lfirst = labels.begin();
        while (dfirst != dlast) {
            if (dfirst + batch_size > dlast) {
                update_weights<U>(dfirst, dlast, lfirst, l_rate, loss_e);
                dfirst = dlast;
            } else {
                update_weights<U>(dfirst, dfirst + batch_size, lfirst, l_rate, loss_e);
                dfirst += batch_size;
                lfirst += batch_size;
            }
        }
    }

    /// Sets the output nonlinearities
    /**
     * Sets the nonlinearities of all nodes connected to the output nodes.
     * This is useful when, for example, the dCGPANN is used for a regression task where output values are expected in
     * [-1 1] and hence the output layer should have some sigmoid or tanh nonlinearity.
     *
     *
     * @param[in] f_id the id of the kernel (nonlinearity)
     *
     * @throw std::invalid_argument if *f_id* is invalid.
     */
    void set_output_f(unsigned f_id)
    {
        for (decltype(this->get_m()) i = 0u; i < this->get_m(); ++i) {
            this->set_f_gene(this->get()[this->get().size() - 1 - i], f_id);
        }
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
    void set_weight(typename std::vector<T>::size_type node_id, typename std::vector<T>::size_type input_id, const T &w)
    {
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_r() * this->get_c()) {
            throw std::invalid_argument("Requested node id does not exist");
        }
        if (input_id >= this->get_arity(node_id)) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }
        // index of the node in the weight vector
        auto idx = this->get_gene_idx()[node_id] - (node_id - this->get_n());
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
     * @throws std::invalid_argument if the input vector dimension is not valid.
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
        if (node_id < this->get_n() || node_id >= this->get_n() + this->get_r() * this->get_c()) {
            throw std::invalid_argument(
                "Requested node id does not exist or does not have a weight (e.g. input nodes)");
        }
        if (input_id >= this->get_arity(node_id)) {
            throw std::invalid_argument("Requested input exceeds the function arity");
        }

        auto idx = this->get_gene_idx()[node_id] - (node_id - this->get_n());
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
     * Gets the values of all the weights.
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
        std::normal_distribution<T> nd{mean, std};
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
#if !defined(DCGP_DOXYGEN_INVOKED)
    void randomise_biases(double mean = 0., double std = 0.1,
                          std::random_device::result_type seed = std::random_device{}())
    {
        std::mt19937 gen{seed};
        std::normal_distribution<T> nd{mean, std};
        for (auto &b : m_biases) {
            b = nd(gen);
        }
    }
#else
    void randomise_biases(double mean = 0, double std = 0.1, std::random_device::result_type seed = random_number) {}
#endif

    /*@}*/

private:
    // For numeric computations
    template <typename U, enable_double<U> = 0>
    U kernel_call(std::vector<U> &function_in, unsigned idx, unsigned node_id, unsigned weight_idx,
                  unsigned bias_idx) const
    {
        // Weights (we transform the inputs a,b,c,d,e in w_1 a, w_2 b, w_3 c, etc...)
        for (auto j = 0u; j < this->get_arity(node_id); ++j) {
            function_in[j] = function_in[j] * m_weights[weight_idx + j];
        }
        // Biases (we add to the first input a bias so that a,b,c,d,e goes in c, etc...))
        function_in[0] += m_biases[bias_idx];
        // We compute the node function that will, for example, map w_1 a + bias, w_2 b, w_3 c,... into f(w_1 a +
        // w_2 b + w_3 c + ... + bias)
        return this->get_f()[this->get()[idx]](function_in);
    }

    // For the symbolic expression
    template <typename U, typename std::enable_if<std::is_same<U, std::string>::value, int>::type = 0>
    U kernel_call(std::vector<U> &function_in, unsigned idx, unsigned node_id, unsigned weight_idx,
                  unsigned bias_idx) const
    {
        // Weights
        for (auto j = 0u; j < this->get_arity(node_id); ++j) {
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
                unsigned arity = this->get_arity(node_id);
                function_in.resize(arity);
                // position in the chromosome of the current node
                unsigned g_idx = this->get_gene_idx()[node_id];
                // starting position in m_weights of the weights relative to the node
                unsigned w_idx = g_idx - (node_id - this->get_n());
                // starting position in m_biases of the node bias
                unsigned b_idx = node_id - this->get_n();
                for (auto j = 0u; j < this->get_arity(node_id); ++j) {
                    function_in[j] = node[this->get()[g_idx + j + 1]];
                }
                node[node_id] = kernel_call(function_in, g_idx, node_id, w_idx, b_idx);
            }
        }
        return node;
    }

    // computes node and node_d to start backprop
    template <typename U, enable_double<U> = 0>
    void fill_nodes(const std::vector<U> &in, std::vector<U> &node, std::vector<U> &d_node) const
    {
        if (in.size() != this->get_n()) {
            throw std::invalid_argument("Input size is incompatible");
        }
        // Start
        std::vector<U> function_in;
        for (auto node_id : this->get_active_nodes()) {
            if (node_id < this->get_n()) {
                node[node_id] = in[node_id];
                // We need d_node to have the same structure of node, hence we also
                // put some bogus entries fot the input nodes that actually do not have an activation function
                // hence no need/use/meaning for a derivative
                d_node[node_id] = 0.;
            } else {
                unsigned arity = this->get_arity(node_id);
                function_in.resize(arity);
                // position in the chromosome of the current node
                unsigned g_idx = this->get_gene_idx()[node_id];
                // starting position in m_weights of the weights relative to the node
                unsigned w_idx = g_idx - (node_id - this->get_n());
                // starting position in m_biases of the node bias
                unsigned b_idx = node_id - this->get_n();
                for (auto j = 0u; j < this->get_arity(node_id); ++j) {
                    function_in[j] = node[this->get()[g_idx + j + 1]];
                }
                node[node_id] = kernel_call(function_in, g_idx, node_id, w_idx, b_idx);
                // take cares of d_node
                // sigmoid derivative is sig(1-sig)
                if (this->get_f()[this->get()[g_idx]].get_name() == "sig") {
                    d_node[node_id] = node[node_id] * (1. - node[node_id]);
                    // tanh derivative is 1 - tanh**2
                } else if (this->get_f()[this->get()[g_idx]].get_name() == "tanh") {
                    d_node[node_id] = 1. - node[node_id] * node[node_id];
                    // Relu derivative is 0 if relu<0, 1 otherwise
                } else if (this->get_f()[this->get()[g_idx]].get_name() == "ReLu") {
                    d_node[node_id] = (node[node_id] > 0.) ? 1. : 0.;
                } else if (this->get_f()[this->get()[g_idx]].get_name() == "ELU") {
                    d_node[node_id] = (node[node_id] > 0.) ? 1. : node[node_id] + 1.;
                } else if (this->get_f()[this->get()[g_idx]].get_name() == "ISRU") {
                    auto cumin = std::accumulate(function_in.begin(), function_in.end(), 0.);
                    d_node[node_id] = node[node_id] * node[node_id] * node[node_id] / cumin / cumin / cumin;
                }
            }
        }
    }

    // This overrides the base class update_data_structures and updates also the m_connected (as well as
    // m_active_nodes and genes). It is called upon construction and each time active genes are changed.
    void update_data_structures()
    {
        expression<T>::update_data_structures();
        m_connected.clear();
        m_connected.resize(this->get_n() + this->get_m() + this->get_r() * this->get_c());
        for (auto node_id : this->get_active_nodes()) {
            if (node_id >= this->get_n()) { // not for input nodes
                // start in the chromosome of the genes expressing the node_id connections
                unsigned idx = this->get_gene_idx()[node_id] + 1u;
                // start in the weight vector of the genes expressing the node_id connections
                unsigned w_idx = (idx - 1u) - (node_id - this->get_n());
                // loop over the genes representing connections
                for (auto i = 0u; i < this->get_arity(node_id); ++i) {
                    if (this->is_active(this->get()[idx + i])) {
                        m_connected[this->get()[idx + i]].push_back(
                            {node_id, w_idx + i});
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
     *
     * @throws std::invalid_argument if the *data* and *label* size do not match or are zero, or if *lr* is not
     * positive.
     */
    template <typename U, enable_double<U> = 0>
    void update_weights(typename std::vector<std::vector<U>>::const_iterator dfirst,
                        typename std::vector<std::vector<U>>::const_iterator dlast,
                        typename std::vector<std::vector<U>>::const_iterator lfirst, U lr, loss_type loss_e)
    {
        U coeff(lr / static_cast<U>(dlast - dfirst));
        while (dfirst != dlast) {
            auto mse_out = d_loss(*dfirst++, *lfirst++, loss_e);
            std::transform(m_weights.begin(), m_weights.end(), std::get<1>(mse_out).begin(), m_weights.begin(),
                           [coeff](U a, U b) { return a - coeff * b; });
            std::transform(m_biases.begin(), m_biases.end(), std::get<2>(mse_out).begin(), m_biases.begin(),
                           [coeff](U a, U b) { return a - coeff * b; });
        }
    }

    template <typename U, enable_double<U> = 0>
    std::tuple<U, std::vector<U>, std::vector<U>> d_loss(typename std::vector<std::vector<U>>::const_iterator dfirst,
                                                         typename std::vector<std::vector<U>>::const_iterator dlast,
                                                         typename std::vector<std::vector<U>>::const_iterator lfirst,
                                                         loss_type loss_e)
    {
        U value(U(0.));
        std::vector<U> gweights(m_weights.size(), U(0.));
        std::vector<U> gbiases(m_biases.size(), U(0.));
        U dim = static_cast<U>(dlast - dfirst);

        while (dfirst != dlast) {
            auto mse_out = d_loss(*dfirst++, *lfirst++, loss_e);
            value += std::get<0>(mse_out);
            std::transform(gweights.begin(), gweights.end(), std::get<1>(mse_out).begin(), gweights.begin(),
                           [dim](U a, U b) { return a + b / dim; });
            std::transform(gbiases.begin(), gbiases.end(), std::get<2>(mse_out).begin(), gbiases.begin(),
                           [dim](U a, U b) { return a + b / dim; });
        }
        value /= dim;

        return std::make_tuple(std::move(value), std::move(gweights), std::move(gbiases));
    }

    double loss(typename std::vector<std::vector<double>>::const_iterator dfirst,
                typename std::vector<std::vector<double>>::const_iterator dlast,
                typename std::vector<std::vector<double>>::const_iterator lfirst, loss_type loss_e)
    {
        double retval(0.);
        double dim = static_cast<double>(dlast - dfirst);
        while (dfirst != dlast) {
            double mse_out = loss(*dfirst++, *lfirst++, loss_e);
            retval += mse_out;
        }
        retval /= dim;

        return retval;
    }

private:
    std::vector<T> m_weights;
    std::vector<std::string> m_weights_symbols;

    std::vector<T> m_biases;
    std::vector<std::string> m_biases_symbols;

    // In order to be able to perform backpropagation on the dCGPANN program, we need to add
    // to the usual CGP data structures one that contains for each node the list of nodes
    // (and weights) it feeds into. We also need to add some virtual nodes (to keep track of output nodes dependencies)
    // The assigned virtual ids starting from n + r * c
    std::vector<std::vector<std::pair<unsigned, unsigned>>> m_connected;
}; // namespace dcgp

} // end of namespace dcgp

#endif // DCGP_EXPRESSION_H

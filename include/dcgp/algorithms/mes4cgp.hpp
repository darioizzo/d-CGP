#ifndef DCGP_MES4CGP_H
#define DCGP_MES4CGP_H

#include <Eigen/Dense>
#include <cmath>
#include <pagmo/algorithm.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <dcgp/problems/symbolic_regression.hpp>
#include <dcgp/rng.hpp>
#include <dcgp/s11n.hpp>

namespace dcgp
{
/// Memetic Evolutionary Strategy for a Cartesian Genetic Program
/**
 *
 * \image html neo-darwinism.jpg "Neo-darwinism"
 *
 * The term Memetic is widely used, in the context of meta-heuristic search, to indicate a synergy between any
 * population-based approach with local improvement procedures. The resulting algorithms are also referred to, in the
 * literature, as Baldwinian evolutionary algorithms (EAs), Lamarckian EAs, cultural algorithms, or genetic local
 * searches. The very same approach, is seen by many just as an hybridization of a global search technique with a
 * local search technique. Regardless of the terminology and point of view, a memetic approach is applicable to symbolic
 * regression tasks and able to improve considerably on the long standing issue of finding constants in
 * Genetic Programming.
 *
 * @see Izzo, Dario, Francesco Biscani, and Alessio Mereta. "Differentiable genetic programming." In European Conference
 * on Genetic Programming, pp. 35-51. Springer, 2017.
 *
 * In this C++ class we offer an UDA (User Defined Algorithm for the pagmo optimization suite) hybridizing the classic
 * Evolutionary Strategy that is traditionally used in Cartesian Genetic Programming research with a second order Newton
 * search step able to help finding the best values for the ephemeral constants. The resulting algorithm is
 * outlined by the following pseudo-algorithm:
 *
 * @code{.unparsed}
 * > Start from a population (pop) of dimension N
 * > while i < gen
 * > > Mutation: create a new population pop2 mutating N times the best individual (only the integer part is affected)
 * > > Life long learning: apply a one step of a second order Newton method to each individual (only the continuous part
 *     is affected)
 * > > Reinsertion: set pop to contain the best N individuals taken from pop and pop2
 * @endcode
 *
 */
class mes4cgp
{
public:
    /// Single entry of the log (gen, fevals, best, constants, formula)
    typedef std::tuple<unsigned, unsigned long long, double, pagmo::vector_double, std::string> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs an evolutionary strategy algorithm for use with a :class:`dcgp::symbolic_regression` UDP.
     *
     * @param gen number of generations.
     * @param max_mut number of active genes to be mutated.
     * @param ftol the algorithm will exit when the loss is below this tolerance.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws std::invalid_argument if *max_mut* is 0 or *ftol* is negative
     */
    mes4cgp(unsigned gen = 1u, unsigned max_mut = 4u, double ftol = 0., unsigned seed = random_device::next())
        : m_gen(gen), m_max_mut(max_mut), m_ftol(ftol), m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        if (max_mut == 0u) {
            throw std::invalid_argument("The number of active mutations is zero, it must be at least 1.");
        }
        if (ftol < 0.) {
            throw std::invalid_argument("The ftol is negative, it must be positive or zero.");
        }
    }

    /// Algorithm evolve method
    /**
     * Evolves the population for a maximum number of generations
     *
     * @param pop population to be evolved
     * @return evolved population
     * @throws std::invalid_argument if a dcgp::symbolic_regression cannot be extracted from the problem
     * @throws std::invalid_argument if the population size is smaller than 2.
     * @throws std::invalid_argument if the number of objectives is not 1.
     */
    pagmo::population evolve(pagmo::population pop) const
    {
        const auto &prob = pop.get_problem();
        auto n_obj = prob.get_nobj();
        const auto bounds = prob.get_bounds();
        auto NP = pop.size();
        auto fevals0 = prob.get_fevals(); // fevals already made
        auto count = 1u;                  // regulates the screen output
        // We do not use directly the pagmo::problem::extract as otherwise we could not override it in the python
        // bindings. Using this global function, instead, allows its implementation to be overridden in the bindings.
        auto udp_ptr = details::extract_sr_cpp_py(prob);
        // PREAMBLE-------------------------------------------------------------------------------------------------
        // Check whether the problem is suitable for mes4cgp
        // If the UDP in pop is not a symbolic_regression UDP, udp_ptr will be NULL
        if (!udp_ptr) {
            throw std::invalid_argument(prob.get_name() + " does not seem to be a symbolic regression problem. "
                                        + get_name()
                                        + " can only be used on problems of the type dcgp::symbolic_regression ");
        }
        if (n_obj > 1) {
            throw std::invalid_argument(prob.get_name() + " has multiple objectives. " + get_name()
                                        + " can only be used on problems that are single objective.");
        }
        if (NP < 2u) {
            throw std::invalid_argument(get_name() + " needs at least 2 individuals in the population, "
                                        + std::to_string(NP) + " detected");
        }
        // Get out if there is nothing to do.
        if (m_gen == 0u) {
            return pop;
        }
        // ---------------------------------------------------------------------------------------------------------

        // No throws, all valid: we clear the logs
        m_log.clear();
        // We make a copy of the cgp which we will use to make mutations.
        auto cgp = udp_ptr->get_cgp();
        // How many ephemeral constants?
        auto n_eph = prob.get_ncx();
        // We get the best chromosome in the population.
        auto best_idx = pop.best_idx();
        auto worst_idx = pop.worst_idx();
        auto best_x = pop.get_x()[best_idx];
        auto best_f = pop.get_f()[best_idx];
        // And we split it into its integer ...
        std::vector<unsigned> best_xu(best_x.size() - n_eph);
        std::transform(best_x.data() + n_eph, best_x.data() + best_x.size(), best_xu.begin(),
                       [](double a) { return boost::numeric_cast<unsigned>(a); });
        // The Hessian sparsity is defined here and it will not change.
        auto hs = prob.hessians_sparsity();
        // (active) Hessian
        Eigen::MatrixXd H;
        // (active) Gradient
        Eigen::MatrixXd G;
        // (active) Constants
        Eigen::MatrixXd C;
        // Full pivoting LU decomposition to check that H is invertible, find the inverse
        // and see if H is positive definite
        Eigen::FullPivLU<Eigen::MatrixXd> fullpivlu;
        // Uniform distribution (to pick the number of active mutations)
        std::uniform_int_distribution<unsigned> dis(1u, m_max_mut);
        // Main loop
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        pagmo::print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                                     "Best:", "\tConstants:", "\tFormula:\n");
                    }
                    auto formula = udp_ptr->prettier(best_x);
                    log_single_line(gen - 1, prob.get_fevals() - fevals0, best_f[0], formula, best_x, n_eph);
                    ++count;
                }
            }
            // 1 - We generate new NP individuals mutating the integer part of the chromosome and leaving the continuous
            // part untouched
            std::vector<pagmo::vector_double> mutated_x(NP, best_x);
            std::vector<pagmo::vector_double> mutated_f(NP, best_f);
            for (decltype(NP) i = 0u; i < NP; ++i) {
                cgp.set(best_xu);
                cgp.mutate_active(dis(m_e));
                std::vector<unsigned> mutated_xu = cgp.get();
                std::transform(mutated_xu.begin(), mutated_xu.end(), mutated_x[i].data() + n_eph,
                               [](unsigned a) { return boost::numeric_cast<double>(a); });
            }

            // 2 - Life long learning is here obtained performing a single Newton iteration (thus favouring constants
            // appearing linearly). The reason this part of the code is very long, is that it has to deal with
            // conversion between the pagmo format and the eigen format for grad and hessians. Furthermore it has to
            // deal with the fact that ephemeral constants are often not appearing in the expression, thus the hessian
            // has zero cols and rows but it still should be used to modify the meaningful constants.
            for (decltype(NP) i = 0u; i < NP; ++i) {
                // Defining the gradient and hessian returned by the UDP.
                auto hess = prob.hessians(mutated_x[i]);
                auto grad = prob.gradient(mutated_x[i]);
                // For a single ephemeral constant we avoid to call the Eigen machinery. This
                // makes the code more readable and results in a few lines rather than tens of.
                if (n_eph == 1u) {
                    if (hess[0][0] != 0. && std::isfinite(grad[0]) & std::isfinite(hess[0][0])) {
                        mutated_x[i][0] = mutated_x[i][0] - grad[0] / hess[0][0];
                    }
                    mutated_f[i] = prob.fitness(mutated_x[i]);
                } else { // We have at least two ephemeral constants defined and thus need to deal with matrices.
                    // We find out how many epheremal constants are actually in expression
                    // collecting the indices of non-zero gradients
                    std::vector<pagmo::vector_double::size_type> non_zero, zero;
                    for (decltype(grad.size()) j = 0u; j < grad.size(); ++j) {
                        if (grad[j] != 0.) {
                            non_zero.emplace_back(j);
                        } else {
                            zero.emplace_back(j);
                        }
                    }
                    auto n_non_zero = non_zero.size();
                    if (n_non_zero == 1u) {
                        // Only one ephemeral constant is in the expression, as above, we avoid to call the
                        // Eigen machinery. This makes the code more readable.
                        if (hess[0][non_zero[0]] != 0. && std::isfinite(grad[non_zero[0]])
                            && std::isfinite(hess[0][non_zero[0]])) {
                            mutated_x[i][non_zero[0]]
                                = mutated_x[i][non_zero[0]] - grad[non_zero[0]] / hess[0][non_zero[0]];
                        }
                    } else if (n_non_zero > 1u) {
                        // The (active) Hessian stored in a square Eigen matrix. Rows and Cols
                        // will be removed later so that size will be n_non_zero x n_non_zero
                        H = Eigen::MatrixXd::Zero(_(n_eph), _(n_eph));
                        // (active) Gradient stored in a column vector
                        G = Eigen::MatrixXd::Zero(_(n_non_zero), 1u);
                        // (active) Constants stored in a column vector
                        C = Eigen::MatrixXd::Zero(_(n_non_zero), 1u);
                        // We construct the hessian including all the zero elements
                        for (decltype(hess[0].size()) j = 0u; j < hess[0].size(); ++j) {
                            H(_(hs[0][j].first), _(hs[0][j].second)) = hess[0][j];
                            H(_(hs[0][j].second), _(hs[0][j].first)) = hess[0][j];
                        }
                        // We remove the rows and columns corresponding to ephemeral constants not
                        // in the expression
                        for (decltype(zero.size()) j = 0u; j < zero.size(); ++j) {
                            remove_row(H, non_zero[j] - j);
                            remove_column(H, non_zero[j] - j);
                        }
                        // We now construct the (active) gradient and the (active) constant column vectors.
                        for (decltype(n_non_zero) j = 0u; j < n_non_zero; ++j) {
                            G(_(j), 0) = grad[non_zero[j]];
                            C(_(j), 0) = mutated_x[i][non_zero[j]];
                        }
                        // And we perform a Newton step
                        if (G.array().isFinite().all()) {
                            fullpivlu.compute(H);
                            // NOTE: the eigenvalues of U (i.e. its diagonal elements) have the same sign structure as
                            // the eigenvalues of H according to Sylvesters theorem of intertia
                            // so we can check that H is positive (semi) definite trough looking at the diagonal
                            // elements of U (see e.g.
                            // https://math.stackexchange.com/questions/621040/compute-eigenvalues-of-a-square-matrix-given-lu-decomposition)
                            if (fullpivlu.isInvertible() && (fullpivlu.matrixLU().diagonal().array() >= 0.).all()) {
                                // H is invertible and positive (semi) definite
                                // we are therefore approaching a minimum and the inverse is defined
                                // it can however contain infinities in some elements
                                // so we check that all elements of the inverse are finite and only then do a Newton
                                // step
                                auto H_inv = fullpivlu.inverse();
                                if (H_inv.array().isFinite().all()) {
                                    C = C - H_inv * G;
                                    // copy modified constants back to the chromosome
                                    for (decltype(non_zero.size()) j = 0u; j < non_zero.size(); ++j) {
                                        mutated_x[i][non_zero[j]] = C(_(j), 0);
                                    }
                                }
                            }
                        }
                    }
                    mutated_f[i] = prob.fitness(mutated_x[i]);
                }
            }
            // 3 - We check if we found anything better.
            for (decltype(NP) i = 0u; i < NP; ++i) {
                if (pagmo::detail::less_than_f(mutated_f[i][0], best_f[0])) {
                    best_f = mutated_f[i];
                    best_x = mutated_x[i];
                    std::copy(mutated_x[i].data() + n_eph, mutated_x[i].data() + mutated_x[i].size(), best_xu.begin());
                }
            }
            // 4 - Exit if ftol is reached
            if (pagmo::detail::greater_than_f(m_ftol, best_f[0])) {
                auto formula = udp_ptr->prettier(best_x);
                log_single_line(gen, prob.get_fevals() - fevals0, best_f[0], formula, best_x, n_eph);
                if (pagmo::detail::less_than_f(best_f[0], pop.get_f()[best_idx][0])) {
                    pop.set_xf(worst_idx, best_x, best_f);
                }
                pagmo::print("Exit condition -- ftol < ", m_ftol, "\n");
                return pop;
            }
        }

        if (pagmo::detail::less_than_f(best_f[0], pop.get_f()[best_idx][0])) {
            pop.set_xf(worst_idx, best_x, best_f);
        }

        if (m_verbosity > 0u) {
            auto formula = udp_ptr->prettier(best_x);
            log_single_line(m_gen, prob.get_fevals() - fevals0, best_f[0], formula, best_x, n_eph);
            pagmo::print("Exit condition -- max generations = ", m_gen, '\n');
        }
        return pop;
    }

    /// Sets the seed
    /**
     * @param seed the seed controlling the algorithm stochastic behaviour
     */
    void set_seed(unsigned seed)
    {
        m_e.seed(seed);
        m_seed = seed;
    }

    /// Gets the seed
    /**
     * @return the seed controlling the algorithm stochastic behaviour
     */
    unsigned get_seed() const
    {
        return m_seed;
    }

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 10):
     * @code{.unparsed}
     *   Gen:        Fevals:          Best:    Constants:   Formula:
     *      0              0        2802.82    [5.35943]    [c1**2] ...
     *     10             40        948.839    [10.9722]    [x0**2*c1] ...
     *     20             80        823.816    [8.38173]    [(c1 + x0)*x0**2] ...
     *     30            120        473.274    [4.48466]    [x0**3*c1] ...
     *     40            160        338.735    [24.2287]    [-x0 + x0**2*c1 - (c1 + x0*c1) + x0**2] ...
     *     50            200        107.126    [24.2287]    [x0**2*(-x0 - x0**2 + x0**3)] ...
     *     60            240        10.2064    [0.844799]   [x0**2*(-(c1 + x0**2) + x0**3)] ...
     *     70            280        10.2064    [0.844799]   [x0**2*(-(c1 + x0**2) + x0**3)] ...
     *     80            320         6.3605    [1.03424]    [x0**2*(x0**3*c1 - (c1 + x0**2*c1))] ...
     *     90            360         6.3605    [1.03424]    [x0**2*(x0**3*c1 - (c1 + x0**2*c1))] ...

     * @endcode
     * Gen is the generation number, Fevals the number of function evaluation used, Best is the best fitness found,
     * Constants contains the value of the ephemeral constants and Formula peeks into the prettier expression of the
     * underlying CGP.
     *
     * @param level verbosity level
     */
    void set_verbosity(unsigned level)
    {
        m_verbosity = level;
    }

    /// Gets the verbosity level
    /**
     * @return the verbosity level
     */
    unsigned get_verbosity() const
    {
        return m_verbosity;
    }

    /// Algorithm name
    /**
     * @return a string containing the algorithm name
     */
    std::string get_name() const
    {
        return "M-ES for CGP: A memetic Evolutionary Strategy for Cartesian Genetic Programming";
    }

    /// Extra info
    /**
     * @return a string containing extra info on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        pagmo::stream(ss, "\tMaximum number of generations: ", m_gen);
        pagmo::stream(ss, "\n\tNumber of active mutations: ", m_max_mut);
        pagmo::stream(ss, "\n\tExit condition of the final loss (ftol): ", m_ftol);
        pagmo::stream(ss, "\n\tVerbosity: ", m_verbosity);
        pagmo::stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
    }

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a mes4cgp::log_line_type containing: Gen, Fevals, Current best, Best as
     * described in mes4cgp::set_verbosity().
     *
     * @return an <tt> std::vector</tt> of mes4cgp::log_line_type containing the logged values Gen, Fevals, Current
     * best, Best
     */
    const log_type &get_log() const
    {
        return m_log;
    }

private:
    static inline void remove_row(Eigen::MatrixXd &matrix, pagmo::vector_double::size_type row_to_remove)
    {
        Eigen::Index n_rows = matrix.rows() - 1u;
        Eigen::Index n_cols = matrix.cols();

        if (_(row_to_remove) < n_rows)
            matrix.block(_(row_to_remove), 0, n_rows - _(row_to_remove), n_cols)
                = matrix.block(_(row_to_remove) + 1, 0, n_rows - _(row_to_remove), n_cols);

        matrix.conservativeResize(n_rows, n_cols);
    }

    static inline void remove_column(Eigen::MatrixXd &matrix, pagmo::vector_double::size_type col_to_remove)
    {
        Eigen::Index n_rows = matrix.rows();
        Eigen::Index n_cols = matrix.cols() - 1u;

        if (_(col_to_remove) < n_cols)
            matrix.block(0, _(col_to_remove), n_rows, n_cols - _(col_to_remove))
                = matrix.block(0, _(col_to_remove) + 1, n_rows, n_cols - _(col_to_remove));

        matrix.conservativeResize(n_rows, n_cols);
    }
    // Eigen stores indexes and sizes as signed types, while dCGP (and pagmo)
    // uses STL containers thus sizes and indexes are unsigned. To
    // make the conversion as painless as possible this template is provided
    // allowing, for example, syntax of the type D(_(i),_(j)) to adress an Eigen matrix
    // when i and j are unsigned
    template <typename I>
    static Eigen::DenseIndex _(I n)
    {
        return static_cast<Eigen::DenseIndex>(n);
    }
    // This prints to screen and logs one single line.
    void log_single_line(unsigned gen, unsigned long long fevals, double best_f, const std::string &formula,
                         const std::vector<double> &best_x, pagmo::vector_double::size_type n_eph) const
    {
        std::vector<double> eph_val(best_x.data(), best_x.data() + n_eph);
        pagmo::print(std::setw(7), gen, std::setw(15), fevals, std::setw(15), best_f, "\t", eph_val, "\t",
                     formula.substr(0, 40) + " ...", '\n');
        m_log.emplace_back(gen, fevals, best_f, eph_val, formula);
    }

public:
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
        ar &m_gen;
        ar &m_max_mut;
        ar &m_ftol;
        ar &m_e;
        ar &m_seed;
        ar &m_verbosity;
        ar &m_log;
    }

private:
    unsigned m_gen;
    unsigned m_max_mut;
    double m_ftol;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};
} // namespace dcgp

PAGMO_S11N_ALGORITHM_EXPORT_KEY(dcgp::mes4cgp)

#endif

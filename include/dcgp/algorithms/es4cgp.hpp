#ifndef DCGP_ES4CGP_H
#define DCGP_ES4CGP_H

#include <boost/optional.hpp>
/* This <boost/serialization/version.hpp> include guards against an issue
 * in boost::serialization from boost 1.74.0 that leads to compiler error
 * "explicit specialization of undeclared template struct 'version'" when
 * including <boost/serialization/optional.hpp>. More details in tickets:
 * https://github.com/boostorg/serialization/issues/210
 * https://github.com/boostorg/serialization/issues/217
 */
#include <boost/serialization/version.hpp>
#include <boost/serialization/optional.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
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
#include <dcgp/std_overloads.hpp>

namespace dcgp
{
/// Evolutionary Strategy for a Cartesian Genetic Program
/**
 *
 * \image html EvolutionaryStrategy.jpg "ES"
 *
 * Evolutionary strategies are popular global optimization meta-heuristics essentially based
 * on the following simple pseudo-algorithm:
 *
 * @code{.unparsed}
 * > Start from a population (pop) of dimension N
 * > while i < gen
 * > > Mutation: create a new population pop2 containing N different mutations of pop best.
 * > > Evaluate all new chromosomes in pop2
 * > > Reinsertion: set pop to contain the best N individuals taken from pop and pop2
 * @endcode
 *
 * The key to the success of such a search strategy is in the quality of its mutation operator. In the
 * case of chrosomoses that encode a Cartesian Genetic Program (CGP), it makes sense to have mutation act
 * on active genes only (that is on that part of the chromosome that is actually expressed in the
 * final CGP / formula / model). This introduces a coupling between the optimization problem (say a symbolic
 * regression problem) and its solution strategy which, although not preventing, makes the use of general purpose
 * optimization algorithms inefficient (e.g. a generic evolutionary strategy would have a mutation operator which
 * is agnostic of the existence of active genes).
 *
 * In this class we provide an evolutionary strategy tailored to solve :class:`dcgp::symbolic_regression` problems
 * leveraging the kowledge on the genetic structure of Cartesian Genetic Programs (i.e. able to mutate only active
 * genes).
 */
class es4cgp
{
public:
    /// Single entry of the log (gen, fevals, best, constants, formula)
    typedef std::tuple<unsigned, unsigned long long, double, pagmo::vector_double, std::string> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs an evolutionary strategy algorithm for use with a cgp::symbolic_regression UDP.
     *
     * @param gen number of generations.
     * @param max_mut maximum number of active genes to be mutated for each individual.
     * @param ftol the algorithm will exit when the loss is below this tolerance. This is useful for cases where
     * an exact formula is seeked, rather than just an approximated one.
     * @param learn_constants when true a gaussian mutation is applied also to the ephemeral constants (std = 0.1).
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws std::invalid_argument if *max_mut* is 0 or *ftol* is negative
     */
    es4cgp(unsigned gen = 1u, unsigned max_mut = 4u, double ftol = 0., bool learn_constants = true,
           unsigned seed = random_device::next())
        : m_gen(gen), m_max_mut(max_mut), m_ftol(ftol), m_learn_constants(learn_constants), 
          m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        if (m_max_mut == 0u) {
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
     * @throws std::invalid_argument if the population size is smaller than 2
     * @throws std::invalid_argument if the number of objectives is not 1.
     */
    pagmo::population evolve(pagmo::population pop) const
    {
        const auto &prob = pop.get_problem();
        auto dim = prob.get_nx();
        auto n_obj = prob.get_nobj();
        const auto bounds = prob.get_bounds();
        auto NP = pop.size();
        auto fevals0 = prob.get_fevals(); // fevals already made
        auto count = 1u;                  // regulates the screen output
        // We do not use directly the pagmo::problem::extract as otherwise we could not override it in the python
        // bindings. Using this global function, instead, allows its implementation to be overridden in the bindings.
        auto udp_ptr = details::extract_sr_cpp_py(prob);
        // PREAMBLE-------------------------------------------------------------------------------------------------
        // Check whether the problem is suitable for es4cgp
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
        auto best_x = pop.get_x()[best_idx];
        double best_f = pop.get_f()[best_idx][0];
        auto mutated_eph_val = std::vector<double>(n_eph, 0.);
        // Normal distribution (to perturb the constants)
        std::normal_distribution<> normal{0., 1.};
        // Uniform distribution (to pick the number of active mutations)
        std::uniform_int_distribution<unsigned> dis(1u, m_max_mut);
        // A contiguous vector of chromosomes/fitness vectors is allocated here
        pagmo::vector_double dvs(NP * dim);
        pagmo::vector_double fs(NP * n_obj);

        // Main loop
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        pagmo::print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                                     "Best:", "\tConstants:", "\tModel:\n");
                    }
                    auto formula = udp_ptr->prettier(best_x);
                    log_single_line(gen - 1, prob.get_fevals() - fevals0, best_f, best_x, formula, n_eph);
                    ++count;
                }
            }
            // 1 - We generate new NP individuals mutating the best and we write on the dvs for pagmo::bfe to evaluate
            // their fitnesses.
            for (decltype(NP) i = 0u; i < NP; ++i) {
                cgp.set_from_range(best_x.begin() + n_eph, best_x.end());
                cgp.mutate_random(dis(m_e));
                std::vector<unsigned> mutated_x = cgp.get();
                std::transform(mutated_x.begin(), mutated_x.end(), dvs.data() + i * dim + n_eph,
                               [](unsigned a) { return boost::numeric_cast<double>(a); });

                // We then mutate the continuous part if requested
                if (m_learn_constants) {
                    for (auto j = 0u; j < n_eph; ++j) {
                        mutated_eph_val[j] = best_x[j] + 10. * normal(m_e);
                    }
                    std::copy(mutated_eph_val.begin(), mutated_eph_val.end(), dvs.data() + i * dim);
                }
            }

            // 3 - We compute the mutants fitnesses
            if (m_bfe) {
                fs = (*m_bfe)(prob, dvs);
            } else {
                for (decltype(NP) i = 0u; i < NP; ++i) {
                    pagmo::vector_double tmp_x(dim);
                    std::copy(dvs.data() + i * dim, dvs.data() + (i + 1) * dim, tmp_x.begin());
                    pagmo::vector_double tmp_f = prob.fitness(tmp_x);
                    fs[i] = tmp_f[0];
                }
            }
            // 4 - We insert the mutated individuals in the population if their fitness is not worse than 
            // the parent's 
            for (decltype(NP) i = 0u; i < NP; ++i) {
                if (!pagmo::detail::greater_than_f(fs[i], best_f)) {
                    best_f = fs[i];
                    // best_x is updated here
                    std::copy(dvs.data() + i * dim, dvs.data() + (i + 1) * dim, best_x.begin());
                }
            }
            // Check if ftol exit condition is met
            if (pagmo::detail::greater_than_f(m_ftol, best_f)) {
                if (m_verbosity > 0u) {
                    auto formula = udp_ptr->prettier(best_x);
                    log_single_line(gen, prob.get_fevals() - fevals0, best_f, best_x, formula, n_eph);
                    ++count;
                    pagmo::print("Exit condition -- ftol < ", m_ftol, "\n");
                }
                update_pop(pop, dvs, fs, best_f, best_x, NP, dim);
                return pop;
            }
        }
        // Evolution has terminated and we now update into the pagmo::pop.
        update_pop(pop, dvs, fs, best_f, best_x, NP, dim);
        // At the end, the pagmo::population will contain the best individual together with its best NP-1 mutants.
        // We log the last iteration
        if (m_verbosity > 0u) {
            auto formula = udp_ptr->prettier(best_x);
            log_single_line(m_gen, prob.get_fevals() - fevals0, best_f, best_x, formula, n_eph);
            pagmo::print("Exit condition -- generations = ", m_gen, '\n');
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

    /// Sets the batch function evaluation scheme
    /**
     * @param b batch function evaluation object
     */
    void set_bfe(const pagmo::bfe &b)
    {
        m_bfe = b;
    }

    /// Sets the algorithm verbosity
    /**
     * Sets the verbosity level of the screen output and of the
     * log returned by get_log(). \p level can be:
     * - 0: no verbosity
     * - >0: will print and log one line each \p level generations.
     *
     * Example (verbosity 100):
     * @code{.unparsed}
     *  Gen:        Fevals:          Best:    Constants:    Model:
     *      0              0        4087.68    [3.52114]    [0] ...
     *    100            400        324.845    [3.61414]    [2*x0**4] ...
     *    200            800        324.845    [3.61414]    [2*x0**4] ...
     *    300           1200        165.212    [3.56702]    [x0**2*(-x0 + 2*x0**2)] ...
     *    400           1600         28.814    [3.45813]    [x0*(-x0 + x0**2*(-x0 + x0**2) - (-x0 +  ...
     *    500           2000        10.5589    [3.59501]    [x0*(-4*x0 + x0**2*(-x0 + x0**2) + x0**2 ...
     *    600           2400         2.2459    [3.44443]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *    700           2800        2.24378    [3.43364]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *    800           3200        2.24378    [3.43364]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *    900           3600        2.24378    [3.43364]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *   1000           4000        2.24374    [3.43618]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *   1100           4400        2.24372    [3.43479]    [x0*(-x0*c1 + x0**2*(-x0 + x0**2) + x0** ...
     *   1200           4800      0.0697188    [3.35616]    [x0*(x0 + x0**2*(-c1 + x0**2))] ...
     *   1300           5200      0.0254527    [3.37625]    [x0*(x0 + x0**2*(-c1 + x0**2))] ...
     * @endcode
     * Gen is the generation number, Fevals the number of function evaluation used, Best is the best fitness found,
     * Constants contains the value of the ephemeral constants and Formula peeks into the prettier expression of the
     * underlying CGP.
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
        return "ES for CGP: Evolutionary strategy for Cartesian Genetic Programming";
    }

    /// Extra info
    /**
     * @return a string containing extra info on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        pagmo::stream(ss, "\tMaximum number of generations: ", m_gen);
        pagmo::stream(ss, "\n\tMaximum number of active mutations: ", m_max_mut);
        pagmo::stream(ss, "\n\tExit condition of the final loss (ftol): ", m_ftol);
        pagmo::stream(ss, "\n\tLearn constants?: ", m_learn_constants);
        pagmo::stream(ss, "\n\tVerbosity: ", m_verbosity);
        pagmo::stream(ss, "\n\tUsing bfe: ", ((m_bfe) ? "yes" : "no"));
        pagmo::stream(ss, "\n\tSeed: ", m_seed);
        return ss.str();
    }

    /// Get log
    /**
     * A log containing relevant quantities monitoring the last call to evolve. Each element of the returned
     * <tt>std::vector</tt> is a es4cgp::log_line_type as described in es4cgp::set_verbosity().
     *
     * @return an <tt> std::vector</tt> of es4cgp::log_line_type containing the logged values.
     */
    const log_type &get_log() const
    {
        return m_log;
    }

private:
    // This prints to screen and logs one single line.
    void log_single_line(unsigned gen, unsigned long long fevals, double best_f, pagmo::vector_double &best_x,
                         const std::string &formula, pagmo::vector_double::size_type n_eph) const
    {
        std::vector<double> eph_val(best_x.data(), best_x.data() + n_eph);
        std::cout << std::setw(7) << gen << std::setw(15) << fevals << std::setw(15) << best_f << "\t" << eph_val
                  << "\t" << formula.substr(0, 40) << " ..." << std::endl;
        m_log.emplace_back(gen, fevals, best_f, eph_val, formula);
    }

    // Used to update the population from the dvs, fs used via the bfe. Typically done at the end of the evolve when an
    // exit condition is met.
    void update_pop(pagmo::population &pop, const pagmo::vector_double &dvs, const pagmo::vector_double &fs,
                    double best_f, const pagmo::vector_double &best_x, pagmo::vector_double::size_type NP,
                    pagmo::vector_double::size_type dim) const
    {
        // First, the latest generation of mutants (if better)
        for (decltype(NP) i = 0u; i < NP; ++i) {
            if (pagmo::detail::less_than_f(fs[i], pop.get_f()[i][0])) {
                pop.set_xf(i, pagmo::vector_double(dvs.data() + i * dim, dvs.data() + (i + 1) * dim),
                           pagmo::vector_double(1, fs[i]));
            }
        }
        auto worst_idx = pop.worst_idx();
        double worst_f_in_pop = pop.get_f()[worst_idx][0];
        // Then the best in place of the worst
        if (pagmo::detail::less_than_f(best_f, worst_f_in_pop)) {
            pop.set_xf(worst_idx, best_x, pagmo::vector_double(1., best_f));
        }
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
        ar &m_learn_constants;
        ar &m_e;
        ar &m_seed;
        ar &m_verbosity;
        ar &m_log;
        ar &m_bfe;
    }

private:
    unsigned m_gen;
    unsigned m_max_mut;
    double m_ftol;
    bool m_learn_constants;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
    boost::optional<pagmo::bfe> m_bfe;
};
} // namespace dcgp

PAGMO_S11N_ALGORITHM_EXPORT_KEY(dcgp::es4cgp)

#endif

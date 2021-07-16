#ifndef DCGP_MOES4CGP_H
#define DCGP_MOES4CGP_H

#include <boost/optional.hpp>
#include <boost/serialization/optional.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/detail/custom_comparisons.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/utils/multi_objective.hpp>
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
/// Multi-Objective, Evolutionary Strategy for a Cartesian Genetic Program
/**
 *
 * \image html multiple_objectives.png "Multi-objective"
 *
 * Symbolic regression tasks seek for good mathematical models to represent input data. By increasing
 * the model complexity it is always (theoretically) possible to find almost perfect fits of any input data.
 * As a consequence, the model complexity must be traded off with its accuracy so that symbolic regression
 * is, ultimately, a two-objectives optimization problem.
 *
 * In this C++ class we offer an UDA (User Defined Algorithm for the pagmo optimization suite) which extends
 * :class:`dcgp::es4cgp` for a multiobjective problem. The resulting approach, is
 * outlined by the following pseudo-algorithm:
 *
 * @code{.unparsed}
 * > Start from a population (pop) of dimension N
 * > while i < gen
 * > > Mutation: create a new population pop2 mutating each individual in pop.
 * > > Reinsertion: set pop to contain the best N individuals taken from pop and pop2 according to non dominated
 * sorting.
 * @endcode
 */
class moes4cgp
{
public:
    /// Single entry of the log (gen, fevals, best loss, ndf size, max. complexity)
    typedef std::tuple<unsigned, unsigned long long, double, unsigned long long, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a multi-objective evolutionary strategy algorithm for use with a
     * :class:`dcgp::symbolic_regression` UDP.
     *
     * @param gen number of generations.
     * @param max_mut maximum number of active genes to be mutated.
     * @param ftol the algorithm will exit when the loss is below this tolerance. This is useful for cases where
     * an exact formula is seeked, rather than just an approximated one.
     * @param learn_constants when true a gaussian mutation is applied to the ephemeral constants (std = 0.1).
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws std::invalid_argument if *max_mut* is 0.
     */
    moes4cgp(unsigned gen = 1u, unsigned max_mut = 4u, double ftol = 0., bool learn_constants = true,
             unsigned seed = random_device::next())
        : m_gen(gen), m_max_mut(max_mut), m_ftol(ftol), m_learn_constants(learn_constants), m_e(seed), m_seed(seed),
          m_verbosity(0u)
    {
        if (max_mut == 0u) {
            throw std::invalid_argument("The maximum number of active mutations is zero, it must be at least 1.");
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
     * @throws std::invalid_argument if a :class:`dcgp::symbolic_regression` cannot be extracted from the problem
     * @throws std::invalid_argument if the population size is smaller than 2.
     * @throws std::invalid_argument if the number of objectives is smaller than 2.
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
        // Check whether the problem is suitable for moes4cgp
        // If the UDP in pop is not a symbolic_regression UDP, udp_ptr will be NULL
        if (!udp_ptr) {
            throw std::invalid_argument(prob.get_name() + " does not seem to be a symbolic regression problem. "
                                        + get_name()
                                        + " can only be used on problems of the type dcgp::symbolic_regression ");
        }
        if (n_obj == 1u) {
            throw std::invalid_argument(get_name()
                                        + " can only be used on multiobjective symbolic regression problems");
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
        // Normal distribution (to perturb the constants)
        std::normal_distribution<> normal{0., 1.};
        // Uniform distribution (to pick the number of active mutations)
        std::uniform_int_distribution<unsigned> dis(1u, m_max_mut);
        // A contiguous vector of chromosomes/fitness vectors for pagmo::bfe input\output is allocated here.
        pagmo::vector_double dvs(NP * dim);
        pagmo::vector_double fs(NP * n_obj);
        // Containers for fitness and decision vectors in a non contiguous structure
        std::vector<pagmo::vector_double> dvs_v(2 * NP, pagmo::vector_double(dim, 0.));
        std::vector<pagmo::vector_double> fs_v(2 * NP, pagmo::vector_double(n_obj, std::numeric_limits<double>::infinity()));
        // This will store the idx of the best individuals to select for the next generation.
        std::vector<pagmo::vector_double::size_type> best_idx(NP);

        // Main loop
        for (decltype(m_gen) gen = 1u; gen <= m_gen; ++gen) {
            // Logs and prints (verbosity modes > 1: a line is added every m_verbosity generations)
            if (m_verbosity > 0u) {
                // Every m_verbosity generations print a log line
                if (gen % m_verbosity == 1u || m_verbosity == 1u) {
                    // Every 50 lines print the column names
                    if (count % 50u == 1u) {
                        pagmo::print("\n", std::setw(7), "Gen:", std::setw(15), "Fevals:", std::setw(15),
                                     "Best loss:", std::setw(10), "Ndf size:", std::setw(10), "Compl.:\n");
                    }
                    log_single_line(gen - 1, prob.get_fevals() - fevals0, pop);
                    ++count;
                    // Check for ftol stopping condition
                    if (pagmo::detail::greater_than_f(m_ftol, pagmo::ideal(pop.get_f())[0])) {
                        if (m_verbosity > 0u) {
                            pagmo::print("Exit condition -- ftol < ", m_ftol, "\n");
                        }
                        return pop;
                    }
                }
            }
            // 1 - We generate new NP individuals mutating the best and we write on the dvs for pagmo::bfe to evaluate
            // their fitnesses.
            for (decltype(NP) i = 0u; i < NP; ++i) {
                cgp.set_from_range(pop.get_x()[i].begin() + n_eph, pop.get_x()[i].end());
                cgp.mutate_random(dis(m_e));
                std::transform(cgp.get().begin(), cgp.get().end(), dvs.data() + i * dim + n_eph,
                               [](unsigned a) { return boost::numeric_cast<double>(a); });

                // We then mutate the continuous part if requested
                if (m_learn_constants) {
                    for (auto j = 0u; j < n_eph; ++j) {
                        dvs[i * dim + j] = pop.get_x()[i][j] + 10. * normal(m_e);
                    }
                }
            }

            // 2 - We compute the mutants fitnesses
            // First we copy the dvs into dvs_v
            for (decltype(NP) i = 0u; i < NP; ++i) {
                std::copy(dvs.data() + i * dim, dvs.data() + (i + 1) * dim, dvs_v[i].begin());
            }
            if (m_bfe) { // bfe evaluation
                fs = (*m_bfe)(prob, dvs);
                for (decltype(NP) i = 0u; i < NP; ++i) {
                    std::copy(fs.data() + i * n_obj, fs.data() + (i + 1) * n_obj, fs_v[i].begin());
                }
            } else { // normal evaluation
                for (decltype(NP) i = 0u; i < NP; ++i) {
                    fs_v[i] = prob.fitness(dvs_v[i]);
                }
            }
            for (decltype(NP) i = 0u; i < NP; ++i) {
                fs_v[i + NP] = pop.get_f()[i];
                dvs_v[i + NP] = pop.get_x()[i];
            }

            // 4 - We select a new population using non dominated sorting
            best_idx = pagmo::select_best_N_mo(fs_v, NP);
            // 5 - We insert into the population
            for (decltype(best_idx.size()) i = 0; i < best_idx.size(); ++i) {
                    pop.set_xf(i, dvs_v[best_idx[i]], fs_v[best_idx[i]]);
            }
        }
        if (m_verbosity > 0u) {
            log_single_line(m_gen, prob.get_fevals() - fevals0, pop);
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
     * Example (verbosity 10):
     * @code{.unparsed}
     *  Gen:        Fevals:     Best loss: Ndf size:   Compl.:
     *     0              0        6.07319         3        92
     *    10           1000        2.15419         5        10
     *    20           2000        1.92403         8        33
     *    30           3000       0.373663        12        72
     *    40           4000        0.36954        13        72
     *    50           5000       0.235749        16        73
     *    60           6000       0.235749        12        73
     *    70           7000       0.235749        13        73
     *    80           8000       0.217968        12        75
     *    90           9000       0.217968        12        75
     *   100          10000       0.217968        12        75
     *   110          11000       0.217968        14        75
     *   120          12000       0.217968        14        75
     *   130          13000       0.217968        13        75
     *   140          14000       0.162293        12        52


     * @endcode
     * Gen is the generation number, Fevals the number of function evaluation used, Best loss is the best loss in
     the
     * population, Ndf size is the size of the non dominated front (i.e. the number of models that are optimal) and
     * compl. is the complexity of the lowest loss model.
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
        return "MO-ES for CGP: Multi-Objective Evolutionary Strategy for Cartesian Genetic Programming";
    }

    /// Extra info
    /**
     * @return a string containing extra info on the algorithm
     */
    std::string get_extra_info() const
    {
        std::ostringstream ss;
        pagmo::stream(ss, "\tNumber of generations: ", m_gen);
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
     * <tt>std::vector</tt> is a moes4cgp::log_line_type containing: Gen, Fevals, Best loss, Ndf size and Complexity
     * described in moes4cgp::set_verbosity().
     *
     * @return an <tt> std::vector</tt> of moes4cgp::log_line_type containing the logged values Gen, Fevals, Best
     * loss, Ndf size and Complexity
     */
    const log_type &get_log() const
    {
        return m_log;
    }

private:
    // This prints to screen and logs one single line.
    void log_single_line(unsigned gen, unsigned long long fevals, const pagmo::population &pop) const
    {
        pagmo::vector_double ideal_point = pagmo::ideal(pop.get_f());
        pagmo::vector_double nadir_point = pagmo::nadir(pop.get_f());
        auto ndf_size = pagmo::non_dominated_front_2d(pop.get_f()).size();
        pagmo::print(std::setw(7), gen, std::setw(15), fevals, std::setw(15), ideal_point[0], std::setw(10), ndf_size,
                     std::setw(10), nadir_point[1], '\n');
        m_log.emplace_back(gen, fevals, ideal_point[0], ndf_size, nadir_point[1]);
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

PAGMO_S11N_ALGORITHM_EXPORT_KEY(dcgp::moes4cgp)

#endif

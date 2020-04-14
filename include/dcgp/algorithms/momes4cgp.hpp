#ifndef DCGP_MOMES4CGP_H
#define DCGP_MOMES4CGP_H

#include <Eigen/Dense>
#include <pagmo/algorithm.hpp>
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
/// Multi-Objective, Memetic Evolutionary Strategy for a Cartesian Genetic Program
/**
 *
 * \image html multiple_objectives.png "Multi-objective"
 * \image html neo-darwinism.jpg "Neo-darwinism"
 *
 * Symbolic regression tasks seek for good mathematical models to represent input data. By increasing
 * the model complexity it is always (theoretically) possible to find almost perfect fits of any input data.
 * As a consequence, the model complexity must be traded off with its accuracy so that symbolic regression
 * is, ultimately, a two-objectives optimization problem.
 *
 * In this C++ class we offer an UDA (User Defined Algorithm for the pagmo optimization suite) which extends
 * :class:`dcgp::mes4cgp` for a multiobjective problem. The resulting algorithm, is
 * outlined by the following pseudo-algorithm:
 *
 * @code{.unparsed}
 * > Start from a population (pop) of dimension N
 * > while i < gen
 * > > Mutation: create a new population pop2 mutating N times the best individual (only the integer part is affected)
 * > > Life long learning: apply a one step of a second order Newton method to each individual (only the continuous part
 *     is affected)
 * > > Reinsertion: set pop to contain the best N individuals taken from pop and pop2 according to non dominated
 * sorting.
 * @endcode
 */
class momes4cgp
{
public:
    /// Single entry of the log (gen, fevals, best loss, ndf size, max. complexity)
    typedef std::tuple<unsigned, unsigned long long, double, unsigned long long, double> log_line_type;
    /// The log
    typedef std::vector<log_line_type> log_type;

    /// Constructor
    /**
     * Constructs a multi-objective memetic evolutionary strategy algorithm for use with a
     * :class:`dcgp::symbolic_regression` UDP.
     *
     * @param gen number of generations.
     * @param max_mut maximum number of active genes to be mutated. The minimum is 0 as to allow multiple steps of
     * Newton descent.
     * @param seed seed used by the internal random number generator (default is random).
     *
     * @throws std::invalid_argument if *mut_n* is 0
     */
    momes4cgp(unsigned gen = 1u, unsigned max_mut = 1u, unsigned seed = random_device::next())
        : m_gen(gen), m_max_mut(max_mut), m_e(seed), m_seed(seed), m_verbosity(0u)
    {
        if (max_mut == 0u) {
            throw std::invalid_argument("The number of active mutations is zero, it must be at least 1.");
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
        auto n_obj = prob.get_nobj();
        const auto bounds = prob.get_bounds();
        auto NP = pop.size();
        auto fevals0 = prob.get_fevals(); // fevals already made
        auto count = 1u;                  // regulates the screen output
        // We do not use directly the pagmo::problem::extract as otherwise we could not override it in the python
        // bindings. Using this global function, instead, allows its implementation to be overridden in the bindings.
        auto udp_ptr = details::extract_sr_cpp_py(prob);
        // PREAMBLE-------------------------------------------------------------------------------------------------
        // Check whether the problem is suitable for momes4cgp
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
        // The hessian will be stored in a square Eigen for inversion
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(_(n_eph), _(n_eph));
        // Gradient and Constants will be stored in these column vectors
        Eigen::MatrixXd G = Eigen::MatrixXd::Zero(_(n_eph), 1);
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(_(n_eph), 1);
        // full pivoting LU decomposition of H to check that H is invertible, to find the inverse and see if H is positive definite
        Eigen::FullPivLU<Eigen::MatrixXd> fullpivlu;
        // auxilary index variables used to construct reduced size Hessian
        unsigned row = 0u;
        unsigned col = 0u;
        auto hs = prob.hessians_sparsity();

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
                }
            }

            // At each generation we need a copy of the population
            pagmo::population popnew(pop);
            // We also need to randomly assign the number of active mutations to each individual. We do this
            // instantiating an n_active_mutations vector:
            std::vector<pagmo::vector_double::size_type> best_idx(NP);
            std::vector<unsigned> n_active_mutations(NP);
            std::iota(n_active_mutations.begin(), n_active_mutations.end(),
                      pagmo::vector_double::size_type(0)); // 0,1,2,3,4,5,6,7,8
            std::transform(
                n_active_mutations.begin(), n_active_mutations.end(), n_active_mutations.begin(),
                [this](pagmo::vector_double::size_type el) { return el % m_max_mut; }); // 0, 1, 2, 3, 0, 1, 2, 3,
            std::shuffle(n_active_mutations.begin(), n_active_mutations.end(), m_e);    // 3, 1, 0, 2, 0, 2, 3, 1,

            // 1 - We generate new NP individuals mutating the integer part of the chromosome and leaving the continuous
            // part untouched
            std::vector<pagmo::vector_double> mutated_x(NP);
            for (decltype(NP) i = 0u; i < NP; ++i) {
                mutated_x[i] = pop.get_x()[i];
                // We extract the integer part of the individual
                std::vector<unsigned> mutated_xu(mutated_x[i].size() - n_eph);
                std::transform(mutated_x[i].data() + n_eph, mutated_x[i].data() + mutated_x[i].size(),
                               mutated_xu.begin(), [](double a) { return boost::numeric_cast<unsigned>(a); });
                // Use it to set the CGP
                cgp.set(mutated_xu);
                // Mutate the expression
                cgp.mutate_active(n_active_mutations[i]);
                mutated_xu = cgp.get();
                // Put it back
                std::transform(mutated_xu.begin(), mutated_xu.end(), mutated_x[i].data() + n_eph,
                               [](unsigned a) { return boost::numeric_cast<double>(a); });
            }

            // 2 - Life long learning (i.e. touching the continuous part) is  obtained performing a single Newton
            // iteration (thus favouring constants appearing linearly)
            for (decltype(NP) i = 0u; i < NP; ++i) {
                pagmo::vector_double grad;
                std::vector<pagmo::vector_double> hess;
                // For a single ephemeral constants we avoid to call the Eigen machinery as its an overkill.
                // I never tested if this is actually also more efficient, but it certainly is more readable.
                if (n_eph == 1u) {
                    hess = prob.hessians(mutated_x[i]);
                    grad = prob.gradient(mutated_x[i]);
                    if (grad[0] != 0.) {
                        mutated_x[i][0] = mutated_x[i][0] - grad[0] / hess[0][0];
                    }
                } else {
                    // One Newton step (NOTE: here we invert an n_eph_active x n_eph_active matrix)
                    // where n_eph_active is the number of epheremal consts that are part of the current expression
                    // We compute hessians and gradients stored in the pagmo format
                    auto hess = prob.hessians(mutated_x[i]);
                    auto grad = prob.gradient(mutated_x[i]);
                    // construct reduced hessian to avoid all zero rows/cols,
                    // since they are linearly dependant and make the matrix non-invertible
                    // find out how many epheremal constants are actually in expression
                    // also collect indices of non-zero gradients
                    std::vector<pagmo::vector_double::size_type> non_zero;
                    for (decltype(grad.size()) j = 0u; j < grad.size(); ++j) {
                        if (grad[j] != 0.){
                            non_zero.emplace_back(j);
                        }
                    }
                    std::vector<pagmo::vector_double::size_type>::size_type n_non_zero = non_zero.size();
                    if (n_non_zero == 1) {
                        // Hessian is a scalar: same as for a single ephemeral constant, except here we take only non-zero gradient
                        if (grad[non_zero[0]] != 0.) {
                            mutated_x[i][non_zero[0]] = mutated_x[i][non_zero[0]] - grad[non_zero[0]] / hess[0][non_zero[0]];
                        }
                    } else if (n_non_zero > 1) {
                        // Hessian is a matrix: use Eigen to invert Hessian
                        // take a part of the original matrices as reduced size Eigen matrices
                        auto H_red = H.topLeftCorner(n_non_zero, n_non_zero);
                        auto G_red = G.topLeftCorner(n_non_zero, 1);
                        auto C_red = C.topLeftCorner(n_non_zero, 1);
                        // copy Hessian, gradient and epheremal constants to Eigen
                        row = 0u;
                        col = 0u;
                        // construct reduced size Hessian without all-zero rows/cols [shape:(n_non_zero, n_non_zero)]
                        for (decltype(hess[0].size()) j = 0u; j < hess[0].size(); ++j) {
                            // hessian is only non-zero where both gradients are non_zero
                            if (std::find(non_zero.begin(), non_zero.end(), hs[0][j].first) != non_zero.end()
                                && std::find(non_zero.begin(), non_zero.end(), hs[0][j].second) != non_zero.end()) {
                                    // we fill the lower diagonal matrix of H_red
                                    H_red(_(row), _(col)) = hess[0][j];
                                    // and mirror to upper
                                    H_red(_(col), _(row)) = hess[0][j];
                                    // sort out if we increment column or row
                                    if (hs[0][j].first == hs[0][j].second) {
                                        // it is a diagonal element
                                        row = row + 1;
                                        col = 0;
                                    } else {
                                        // off-diagonal, only increase columns count
                                        col = col + 1;
                                    }
                            }
                        }
                        // reduced size gradient and epheremal constant vectors [shape:(n_non_zero, 1)]
                        for (decltype(n_non_zero) j = 0u; j < n_non_zero; ++j) {
                            C_red(_(j), 0) = mutated_x[i][non_zero[j]];
                            G_red(_(j), 0) = grad[non_zero[j]];
                        }
                        // make sure that gradients are finite
                        if (G_red.array().isFinite().all()) {
                            // TODO: should we log/print what we decided to do?, i.e. Newton step or not?
                            // compute full pivoting LU decomposition
                            // NOTE: the eigenvalues of U (i.e. its diagonal elements) have the same sign structure as
                            // the eigenvalues of H_red according to sylvesters theorem of intertia
                            // so we can check that H_red is positive (semi) definite trough looking at the diagonal elements of U
                            // (see e.g. https://math.stackexchange.com/questions/621040/compute-eigenvalues-of-a-square-matrix-given-lu-decomposition)
                            fullpivlu.compute(H_red);
                            if (fullpivlu.isInvertible() && (fullpivlu.matrixLU().diagonal().array() >= 0.).all()) {
                                // H_red is invertible and positive (semi) definite
                                // we are therefore approaching a minimum and the inverse is defined
                                // it can however contain infinities in some elements
                                // so we check that all elements of the inverse are finite and only then do a Newton step
                                auto inv = fullpivlu.inverse();
                                if (inv.array().isFinite().all()) {
                                    C_red = C_red - fullpivlu.inverse() * G_red;
                                    // copy modified constants back to pagmo (only if we modified)
                                    for (decltype(non_zero.size()) j = 0u; j < non_zero.size(); ++j) {
                                        mutated_x[i][non_zero[j]] = C(_(j), 0);
                                    }
                                }
                            }
                        }
                    }
                }
                // We use prob to evaluate the fitness so its feval counter is increased.
                auto f = prob.fitness(mutated_x[i]);
                // Diversity mechanism. If the fitness is already present we do not insert the individual.
                // Do I need this copy? @bluescarni? Can I use the get in the find directly? its a ref I think
                // so yes in theory....
                auto fs = popnew.get_f();
                if (std::find(fs.begin(), fs.end(), f) == fs.end() && std::isfinite(f[0])) {
                    popnew.push_back(mutated_x[i], f);
                }
            }
            // 3 - We select a new population using non dominated sorting
            best_idx = pagmo::select_best_N_mo(popnew.get_f(), NP);
            // We insert into the population
            for (pagmo::population::size_type i = 0; i < NP; ++i) {
                pop.set_xf(i, popnew.get_x()[best_idx[i]], popnew.get_f()[best_idx[i]]);
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
     * Gen is the generation number, Fevals the number of function evaluation used, Best loss is the best loss in the
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
        return "MOM-ES for CGP: MultiObjective Memetic Evolutionary Strategy for Cartesian Genetic Programming";
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
    template <typename Archive>
    void serialize(Archive &ar, unsigned)
    {
        ar &m_gen;
        ar &m_max_mut;
        ar &m_e;
        ar &m_seed;
        ar &m_verbosity;
        ar &m_log;
    }

private:
    unsigned m_gen;
    unsigned m_max_mut;
    mutable detail::random_engine_type m_e;
    unsigned m_seed;
    unsigned m_verbosity;
    mutable log_type m_log;
};
} // namespace dcgp

PAGMO_S11N_ALGORITHM_EXPORT_KEY(dcgp::momes4cgp)

#endif

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <pagmo/algorithm.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <symengine/expression.h>
#include <vector>

#include <dcgp/algorithms/moes4cgp.hpp>
#include <dcgp/gym.hpp>
#include <dcgp/kernel_set.hpp>
#include <dcgp/problems/symbolic_regression.hpp>

namespace po = boost::program_options;
using namespace dcgp;
using namespace boost::algorithm;

int main(int ac, char *av[])
{
    // PREAMBLE ----------------------------------------------------------------
    // We deal with the command line options as to create a nicely commented executable
    unsigned gen, max_mut, n_eph, pop_size, verbosity;
    bool bfe;
    po::options_description desc("Allowed options");
    desc.add_options()("help, h", "produce help message")(
        "generations, g", po::value<unsigned>(&gen)->default_value(10000u), "number of generations")(
        "max_mut, m", po::value<unsigned>(&max_mut)->default_value(15u), "maximum number of mutations allowed")(
        "n_eph, c", po::value<unsigned>(&n_eph)->default_value(3u), "number of constants in the expression")(
        "verbosity, v", po::value<unsigned>(&verbosity)->default_value(500u), "the screen frequency")(
        "bfe, b", po::bool_switch(&bfe), "activates the parallel batch evaluator (for large populations)")(
        "pop_size, p", po::value<unsigned>(&pop_size)->default_value(4u), "population size");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    // ---------------------------------------------------------------------------

    // We load the data
    std::vector<std::vector<double>> X, Y;
    gym::generate_P1(X, Y);
    // We instantiate a symbolic regression problem with no ephemeral constants.
    symbolic_regression udp(X, Y, 1u, 16u, 17u, 2u, kernel_set<double>({"sum", "diff", "mul", "pdiv"})(), n_eph, true,
                            0u, "MSE");

    // We init the population
    pagmo::population pop{udp, pop_size};

    for (decltype(pop_size) i = 0u; i < pop_size; ++i) {
        auto prettier = udp.prettier(pop.get_x()[i]);
        trim_left_if(prettier, is_any_of("["));
        trim_right_if(prettier, is_any_of("]"));
        pagmo::print(std::setw(2), i + 1, " - Loss: ", std::setw(13), std::left, pop.get_f()[i][0], std::setw(15),
                     "Complexity: ", std::left, std::setw(5), pop.get_f()[i][1], std::setw(10), "Formula: ", prettier,
                     "\n");
    }

    // And we define an evolutionary startegy
    dcgp::moes4cgp uda(gen, max_mut, 1e-8, true);
    if (bfe) {
        uda.set_bfe(pagmo::bfe{});
    }
    pagmo::algorithm algo{uda};
    algo.set_verbosity(verbosity);

    // We evolve the population
    pop = algo.evolve(pop);

    // Finally we print on screen the non dominated front.
    pagmo::print("\nNon dominated Front at the end:\n");
    auto ndf = pagmo::non_dominated_front_2d(pop.get_f());
    for (decltype(ndf.size()) i = 0u; i < ndf.size(); ++i) {
        auto idx = ndf[i];
        auto prettier = udp.prettier(pop.get_x()[idx]);
        trim_left_if(prettier, is_any_of("["));
        trim_right_if(prettier, is_any_of("]"));
        pagmo::print(std::setw(2), i + 1, " - Loss: ", std::setw(13), std::left, pop.get_f()[idx][0], std::setw(15),
                     "Complexity: ", std::left, std::setw(5), pop.get_f()[idx][1], std::setw(10), "Formula: ", prettier,
                     "\n");
    }

    return false;
}

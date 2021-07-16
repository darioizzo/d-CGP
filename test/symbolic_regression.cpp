#define BOOST_TEST_MODULE dcgp_symbolic_regression_test
#include <boost/test/included/unit_test.hpp>

#include <sstream>

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/io.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

#include <dcgp/gym.hpp>
#include <dcgp/problems/symbolic_regression.hpp>
#include <dcgp/s11n.hpp>
#include <dcgp/wrapped_functions_s11n_implement.hpp>

using namespace dcgp;

BOOST_AUTO_TEST_CASE(construction_test)
{
    // Its default-onstructable
    BOOST_CHECK_NO_THROW(symbolic_regression{});
    // Sanity checks tests (inconsistent points / labels)
    BOOST_CHECK_THROW(symbolic_regression({}, {}), std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}, {0.02 / 0.32}}),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32, 0.3}}, {{3. / 2.}, {0.02 / 0.32}}),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2., 2.2}, {0.02 / 0.32}}),
                      std::invalid_argument);
    // Sanity checks tests (inconsistent cgp parameters)
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 0u, 1u, 1u, 2u,
                                          kernel_set<double>({"sum"})(), 0u, false),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 1u, 0u, 1u, 2u,
                                          kernel_set<double>({"sum"})(), 0u, false),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 1u, 1u, 0u, 2u,
                                          kernel_set<double>({"sum"})(), 0u, false),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 1u, 1u, 1u, 1u,
                                          kernel_set<double>({"sum"})(), 0u, false),
                      std::invalid_argument);
    BOOST_CHECK_THROW(symbolic_regression({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 1u, 1u, 1u, 2u,
                                          kernel_set<double>(std::vector<std::string>())(), 0u, false),
                      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(fitness_test_single_obj)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_uball5d(points, labels);
    // 2xy, 2x
    pagmo::vector_double test_x = {0, 1, 1, 0, 0, 0, 2, 0, 2, 2, 0, 2, 4, 3};
    // On a single point/label.
    {
        symbolic_regression udp({{1., 1.}}, {{2., 2.}}, 2, 2, 3, 2, basic_set(), 0u, false, 0u);
        BOOST_CHECK_EQUAL(udp.fitness(test_x)[0], 0.);
    }
    {
        symbolic_regression udp({{1., 1.}}, {{0., 0.}}, 2, 2, 3, 2, basic_set(), 0u, false, 0u);
        BOOST_CHECK_EQUAL(udp.fitness(test_x)[0], 4.);
    }
    {
        symbolic_regression udp({{1., 0.}}, {{0., 0.}}, 2, 2, 3, 2, basic_set(), 0u, false, 0u);
        BOOST_CHECK_EQUAL(udp.fitness(test_x)[0], 2.);
    }
    // On a batch (first sequential then parallel)
    {
        symbolic_regression udp({{1., 1.}, {1., 0.}}, {{2., 2.}, {0., 0.}}, 2, 2, 3, 2, basic_set(), 0u, false, 0u);
        BOOST_CHECK_EQUAL(udp.fitness(test_x)[0], 1.);
    }
    {
        symbolic_regression udp({{1., 1.}, {1., 0.}}, {{2., 2.}, {0., 0.}}, 2, 2, 3, 2, basic_set(), 0u, false, 1u);
        BOOST_CHECK_EQUAL(udp.fitness(test_x)[0], 1.);
    }
    // c1-c2-x, c1+2y
    pagmo::vector_double test_xeph
        = {1., 2., 0, 0, 2, 1, 0, 1, 1, 2, 3, 0, 3, 1, 1, 6, 0, 0, 4, 1, 2, 1, 1, 1, 9, 5, 2, 3, 3, 0, 5, 0, 8, 11};
    {
        symbolic_regression udp({{1., 0.}}, {{0., 3.}}, 1, 10, 11, 2, basic_set(), 2u, false, 1u);
        // 1 - 2 - 1, 1
        BOOST_CHECK_EQUAL(udp.fitness(test_xeph)[0], 4.);
    }
    {
        symbolic_regression udp({{1., 0.}}, {{-2., 1.}}, 1, 10, 11, 2, basic_set(), 2u, false, 1u);
        // 1 - 2 - 1, 1
        BOOST_CHECK_EQUAL(udp.fitness(test_xeph)[0], 0.);
    }
    {
        symbolic_regression udp({{-1, -1}}, {{0., -1.}}, 1, 10, 11, 2, basic_set(), 2u, false, 1u);
        // 1 - 2 + 1, 1 - 2
        BOOST_CHECK_EQUAL(udp.fitness(test_xeph)[0], 0.);
        // 1 - 2 + 1, 1 - 2
        test_xeph[0] = 1.;
        test_xeph[1] = 2.;
        BOOST_CHECK_EQUAL(udp.fitness(test_xeph)[0], 0.);
        // 3 - 3 + 1, 3 - 2
        test_xeph[0] = 3.;
        test_xeph[1] = 3.;
        BOOST_CHECK_EQUAL(udp.fitness(test_xeph)[0], 2.5);
    }
}
BOOST_AUTO_TEST_CASE(fitness_test_two_obj)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_uball5d(points, labels);
    symbolic_regression udp{points, labels, 1, 15, 16, 2, basic_set(), 2, true, 0u};
    pagmo::population pop(udp, 10u);
    for (decltype(pop.size()) i = 0u; i < pop.size(); ++i) {
        udp.set_cgp(pop.get_x()[i]);
        BOOST_CHECK_EQUAL(udp.get_cgp().get_active_genes().size(), pop.get_f()[i][1]);
    }
}


BOOST_AUTO_TEST_CASE(get_bounds_test)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    symbolic_regression udp({{1., 2.}, {0.3, -0.32}}, {{3. / 2.}, {0.02 / 0.32}}, 2, 2, 3, 2, basic_set(), 0u, 1u);
    expression<double> cgp(2, 1, 2, 2, 3, 2, basic_set(), 0u, 23u);
    auto lbu = cgp.get_lb();
    std::vector<double> lb(lbu.size());
    std::transform(lbu.begin(), lbu.end(), lb.begin(), [](unsigned a) { return boost::numeric_cast<double>(a); });
    BOOST_CHECK(udp.get_bounds().first == lb);
}

BOOST_AUTO_TEST_CASE(trivial_methods_test)
{
    symbolic_regression udp({{1., 1.}}, {{2., 2.}}, 2, 2, 3, 2);
    BOOST_CHECK_EQUAL(udp.get_bounds().first.size(), udp.get_nix());
    BOOST_CHECK(udp.get_name().find("CGP") != std::string::npos);
    BOOST_CHECK(udp.get_extra_info().find("Data dimension") != std::string::npos);
    pagmo::vector_double test_x = {0, 1, 1, 0, 0, 0, 2, 0, 2, 2, 0, 2, 4, 3};
    BOOST_CHECK(udp.pretty(test_x).find("[(x0*(x1+x1)), (x0+x0)]") != std::string::npos);
    BOOST_CHECK(udp.prettier(test_x).find("[2*x0*x1, 2*x0]") != std::string::npos);
    BOOST_CHECK_NO_THROW(udp.get_cgp());
}

BOOST_AUTO_TEST_CASE(gradient_test)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_koza_quintic(points, labels);
    symbolic_regression udp(points, labels, 5, 10, 3, 2, basic_set(), 5u, 0u);
    BOOST_CHECK_EQUAL(udp.gradient_sparsity().size(), 5u);
    BOOST_CHECK((udp.gradient_sparsity() == pagmo::sparsity_pattern{{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}}));
    // Here we test (a bit arbitrarily) that the gradient points in a direction of decreasing the loss.
    for (unsigned j = 0u; j < 10u; ++j) {
        pagmo::population pop(udp, 1u, 123u + j);
        auto x = pop.get_x()[0];
        auto f1 = udp.fitness(x);
        auto g1 = udp.gradient(x);
        double loss_gradient_norm = std::sqrt(std::inner_product(g1.begin(), g1.end(), g1.begin(), 0.));
        for (unsigned i = 0u; i < 5u; ++i) {
            x[i] = x[i] - 1e-8 * g1[i] / loss_gradient_norm;
        }
        if (std::isfinite(f1[0]) && std::isfinite(loss_gradient_norm) && (loss_gradient_norm != 0.)) {
            BOOST_CHECK(f1[0] - udp.fitness(x)[0] >= 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(hessians_test)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_koza_quintic(points, labels);
    // n_eph = 0
    {
        symbolic_regression udp(points, labels, 5, 10, 3, 2, basic_set(), 0u, 0u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity().size(), 1u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity()[0].size(), 0u);
        BOOST_CHECK((udp.hessians_sparsity() == std::vector<pagmo::sparsity_pattern>{{}}));
    }
    // n_eph = 3
    {
        symbolic_regression udp(points, labels, 5, 10, 3, 2, basic_set(), 3u, 0u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity().size(), 1u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity()[0].size(), 6u);
        BOOST_CHECK((udp.hessians_sparsity()
                     == std::vector<pagmo::sparsity_pattern>{{{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}}}));
    }
    // n_eph = 10
    {
        symbolic_regression udp(points, labels, 5, 10, 3, 2, basic_set(), 10u, 0u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity().size(), 1u);
        BOOST_CHECK_EQUAL(udp.hessians_sparsity()[0].size(), 55u);
    }
    // 2 - We test the numerical value over a simple expression
    // c1-c2-x, c1+2y
    pagmo::vector_double test_xeph
        = {1.23, 2.34, 0, 0, 2, 1, 0, 1, 1, 2, 3, 0, 3, 1, 1, 6, 0, 0, 4, 1, 2, 1, 1, 1, 9, 5, 2, 3, 3, 0, 5, 0, 8, 11};
    symbolic_regression udp({{1., 0.}}, {{0., 3.}}, 1, 10, 11, 2, basic_set(), 2u, false);
    BOOST_CHECK(udp.hessians(test_xeph) == std::vector<pagmo::vector_double>(1, pagmo::vector_double{2, -1, 1}));
}

BOOST_AUTO_TEST_CASE(cache_test)
{
    // NOTE: this is not testing whether the cache is hit, but assuming it is
    // it tests that it returns the correct result.
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_koza_quintic(points, labels);
    symbolic_regression udp(points, labels, 2, 2, 3, 2, basic_set(), 5u, 0u);
    pagmo::population pop(udp, 1u);
    // case 1: cache written by gradient
    {
        auto f1 = udp.fitness(pop.get_x()[0]);
        auto g1 = udp.gradient(pop.get_x()[0]);
        auto f2 = udp.fitness(pop.get_x()[0]);
        BOOST_CHECK_CLOSE(f1[0], f2[0], 1e-12);
    }
    // case 2: cache written by hessians
    {
        auto f1 = udp.fitness(pop.get_x()[0]);
        auto g1 = udp.gradient(pop.get_x()[0]);
        auto h1 = udp.hessians(pop.get_x()[0]);
        auto g2 = udp.gradient(pop.get_x()[0]);
        auto f2 = udp.fitness(pop.get_x()[0]);
        BOOST_CHECK_CLOSE(f1[0], f2[0], 1e-12);
        BOOST_CHECK(g1 == g2);
    }
}

BOOST_AUTO_TEST_CASE(s11n_test)
{
    kernel_set<double> basic_set({"sum", "diff", "mul", "div"});
    std::vector<std::vector<double>> points, labels;
    gym::generate_koza_quintic(points, labels);
    symbolic_regression udp(points, labels, 2, 2, 3, 2, basic_set(), 5u, 0u);

    const auto orig = udp.get_extra_info();

    std::stringstream ss;
    {
        boost::archive::binary_oarchive oarchive(ss);
        oarchive << udp;
    }
    udp = symbolic_regression(points, labels, 2, 2, 3, 2, kernel_set<double>({"sum", "diff"})(), 5u, 0u);
    {
        boost::archive::binary_iarchive iarchive(ss);
        iarchive >> udp;
    }

    BOOST_CHECK(orig == udp.get_extra_info());
}

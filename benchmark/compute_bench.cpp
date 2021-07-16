#include <audi/audi.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>

#include <dcgp/expression.hpp>
#include <dcgp/kernel_set.hpp>

using gdual_d = audi::gdual_d;

void perform_evaluations(unsigned int in, unsigned int out, unsigned int rows, unsigned int columns,
                         unsigned int levels_back, unsigned int arity, unsigned int N,
                         std::vector<dcgp::kernel<double>> kernel_set)
{
    // Random numbers engine
    std::default_random_engine re(123);
    // Instatiate the expression
    dcgp::expression<double> ex(in, out, rows, columns, levels_back, arity, kernel_set, 0u, 123u);
    // We create the input data upfront and we do not time it.
    std::vector<double> dumb(in);
    std::vector<std::vector<double>> in_num(N, dumb);

    for (auto j = 0u; j < N; ++j) {
        for (auto i = 0u; i < in; ++i) {
            in_num[j][i] = std::uniform_real_distribution<double>(-1, 1)(re);
        }
    }

    std::cout << "Performing " << N << " evaluations, in:" << in << " out:" << out << " rows:" << rows
              << " columns:" << columns << std::endl;
    {
        boost::timer::auto_cpu_timer t;
        for (auto i = 0u; i < N; ++i) {
            ex(in_num[i]);
        }
    }
}

/// This torture test is passed whenever it completes. It is meant to check for
/// the code stability when large number of mutations are performed
int main()
{
    unsigned int N = 100000;

    dcgp::kernel_set<double> kernel_set1({"sum", "diff", "mul", "div"});
    audi::stream(std::cout, "Function set ", kernel_set1(), "\n");
    perform_evaluations(1, 1, 1, 16, 17, 2, N, kernel_set1());
    perform_evaluations(2, 4, 2, 3, 4, 4, N, kernel_set1());
    perform_evaluations(2, 4, 10, 10, 11, 5, N, kernel_set1());
    perform_evaluations(2, 4, 20, 20, 21, 6, N, kernel_set1());
    perform_evaluations(1, 1, 1, 100, 101, 7, N, kernel_set1());
    perform_evaluations(1, 1, 2, 100, 101, 8, N, kernel_set1());
    perform_evaluations(1, 1, 3, 100, 101, 9, N, kernel_set1());

    dcgp::kernel_set<double> kernel_set2({"sum", "mul", "sig"});
    audi::stream(std::cout, "\nFunction set ", kernel_set2(), "\n");
    perform_evaluations(2, 4, 2, 3, 4, 4, N, kernel_set2());
    perform_evaluations(2, 4, 10, 10, 11, 5, N, kernel_set2());
    perform_evaluations(2, 4, 20, 20, 21, 6, N, kernel_set2());
    perform_evaluations(1, 1, 1, 100, 101, 7, N, kernel_set2());
    perform_evaluations(1, 1, 2, 100, 101, 8, N, kernel_set2());
    perform_evaluations(1, 1, 3, 100, 101, 9, N, kernel_set2());
    return 0;
}

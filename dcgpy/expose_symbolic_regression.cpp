#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>

#include <dcgp/algorithms/es4cgp.hpp>
#include <dcgp/algorithms/gd4cgp.hpp>
#include <dcgp/algorithms/mes4cgp.hpp>
#include <dcgp/algorithms/moes4cgp.hpp>
#include <dcgp/algorithms/momes4cgp.hpp>
#include <dcgp/gym.hpp>
#include <dcgp/kernel.hpp>
#include <dcgp/problems/symbolic_regression.hpp>

#include <pagmo/algorithm.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "common_utils.hpp"
#include "docstrings.hpp"

PAGMO_S11N_ALGORITHM_IMPLEMENT(dcgp::es4cgp)
PAGMO_S11N_ALGORITHM_IMPLEMENT(dcgp::mes4cgp)
PAGMO_S11N_ALGORITHM_IMPLEMENT(dcgp::moes4cgp)
PAGMO_S11N_ALGORITHM_IMPLEMENT(dcgp::momes4cgp)
PAGMO_S11N_ALGORITHM_IMPLEMENT(dcgp::gd4cgp)
PAGMO_S11N_PROBLEM_IMPLEMENT(dcgp::symbolic_regression)

namespace py = pybind11;
using namespace dcgp;
using namespace dcgpy;

using gym_ptr = void (*)(std::vector<std::vector<double>> &, std::vector<std::vector<double>> &);
template <gym_ptr F>
inline void expose_data_from_the_gym(py::module &m, const std::string &name, const std::string &docstring)
{
    m.def(
        name.c_str(),
        []() {
            std::vector<std::vector<double>> points, labels;
            F(points, labels);
            return py::make_tuple(dcgpy::vvector_to_ndarr(points), dcgpy::vvector_to_ndarr(labels));
        },
        docstring.c_str());
}

std::vector<std::vector<double>> to_vv(const py::array_t<double> &in)
{
    auto vv = dcgpy::ndarr_to_vvector(in);
    return vv;
}

py::array_t<double> from_vv(const std::vector<std::vector<double>> &in)
{
    auto arr = dcgpy::vvector_to_ndarr(in);
    return arr;
}

namespace dcgpy
{
void expose_symbolic_regression(py::module &m)
{
    m.def("to_vv", &to_vv);
    m.def("from_vv", &from_vv);

    // We expose the UDPs
    py::class_<dcgp::symbolic_regression> sr_(m, "symbolic_regression", symbolic_regression_doc().c_str());
    sr_.def(py::init<>())
        // Constructor from list of lists
        .def(py::init<const std::vector<std::vector<double>> &, const std::vector<std::vector<double>> &, unsigned,
                      unsigned, unsigned, unsigned, const std::vector<kernel<double>> &, unsigned, bool, unsigned,
                      std::string>(),
             py::arg("points"), py::arg("labels"), py::arg("rows") = 1, py::arg("cols") = 16,
             py::arg("levels_back") = 17, py::arg("arity") = 2, py::arg("kernels"), py::arg("n_eph") = 0u,
             py::arg("multi_objective") = false, py::arg("parallel_batches") = 0u, py::arg("loss") = "MSE")
        // Constructor from Numpy Arrays
        .def(
            py::init([](const py::array_t<double> &points, const py::array_t<double> &labels, unsigned rows,
                        unsigned cols, unsigned levels_back, unsigned arity, const std::vector<kernel<double>> &kernels,
                        unsigned n_eph, bool multi_objective, unsigned parallel_batches, std::string loss) {
                auto vvd_points = ndarr_to_vvector(points);
                auto vvd_labels = ndarr_to_vvector(labels);
                return ::new dcgp::symbolic_regression(vvd_points, vvd_labels, rows, cols, levels_back, arity, kernels,
                                                       n_eph, multi_objective, parallel_batches, loss);
            }),
            symbolic_regression_init_doc().c_str(), py::arg("points"), py::arg("labels"), py::arg("rows") = 1,
            py::arg("cols") = 16, py::arg("levels_back") = 17, py::arg("arity") = 2, py::arg("kernels"),
            py::arg("n_eph") = 0u, py::arg("multi_objective") = false, py::arg("parallel_batches") = 0u,
            py::arg("loss") = "MSE")
        .def("get_nobj", &dcgp::symbolic_regression::get_nobj)
        .def("get_cgp", &dcgp::symbolic_regression::get_cgp)
        .def("fitness", &dcgp::symbolic_regression::fitness)
        .def("gradient", &dcgp::symbolic_regression::gradient)
        .def("gradient_sparsity",
             [](const dcgp::symbolic_regression &instance) { return sp_to_ndarr(instance.gradient_sparsity()); })
        .def("hessians", &dcgp::symbolic_regression::hessians)
        .def("hessians_sparsity",
             [](const dcgp::symbolic_regression &instance) {
                 auto hs = instance.hessians_sparsity();
                 std::vector<py::array_t<pagmo::vector_double::size_type>> retval;
                 for (const auto &h : hs) {
                     retval.push_back(sp_to_ndarr(h));
                 }
                 return retval;
             })
        .def("get_bounds", &dcgp::symbolic_regression::get_bounds)
        .def("get_nix", &dcgp::symbolic_regression::get_nix)
        .def("get_name", &dcgp::symbolic_regression::get_name)
        .def("get_extra_info", &dcgp::symbolic_regression::get_extra_info)
        .def("pretty", &dcgp::symbolic_regression::pretty)
        .def("prettier", &dcgp::symbolic_regression::prettier)
        .def(
            "predict",
            [](const dcgp::symbolic_regression &instance, const py::array_t<double> &points,
               const std::vector<double> &x) {
                try { // We first try with 2D array, if not we assume its a 1D array and try.
                    return vvector_to_ndarr(instance.predict(ndarr_to_vvector(points), x));
                } catch (...) {
                    PyErr_Clear();
                    return vector_to_ndarr(instance.predict(ndarr_to_vector(points), x));
                }
            },
            py::arg("points"), py::arg("chromosome"), symbolic_regression_predict_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::symbolic_regression>,
                        &udx_pickle_setstate<dcgp::symbolic_regression>))
        .def("__repr__", &dcgp::symbolic_regression::get_extra_info);

    // We expose the UDAs
    // ES-4CGP (Evolutionary Strategy for Cartesian Genetic Programming)
    py::class_<dcgp::es4cgp> es4cgp_(m, "es4cgp", es4cgp_doc().c_str());
    es4cgp_
        .def(py::init<unsigned, unsigned, double, bool>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("learn_constants") = true)
        .def(py::init<unsigned, unsigned, double, bool, unsigned>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("learn_constants") = true, py::arg("seed"))
        .def("evolve", &dcgp::es4cgp::evolve)
        .def("set_verbosity", &dcgp::es4cgp::set_verbosity)
        .def("get_name", &dcgp::es4cgp::get_name)
        .def("get_extra_info", &dcgp::es4cgp::get_extra_info)
        .def("get_seed", &dcgp::es4cgp::get_seed, generic_uda_get_seed_doc().c_str())
        .def("set_bfe", &dcgp::es4cgp::set_bfe, generic_set_bfe_doc().c_str(), py::arg("b"))
        .def("get_log", &generic_log_getter<dcgp::es4cgp>, es4cgp_get_log_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::es4cgp>, &udx_pickle_setstate<dcgp::es4cgp>))
        .def("__repr__", &dcgp::es4cgp::get_extra_info);
    // MOES-4CGP (Multi-Objective Evolutionary Strategy for Cartesian Genetic Programming)
    py::class_<dcgp::moes4cgp> moes4cgp_(m, "moes4cgp", moes4cgp_doc().c_str());
    moes4cgp_
        .def(py::init<unsigned, unsigned, double, bool>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("learn_constants") = true)
        .def(py::init<unsigned, unsigned, double, bool, unsigned>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("learn_constants") = true, py::arg("seed"))
        .def("evolve", &dcgp::moes4cgp::evolve)
        .def("set_verbosity", &dcgp::moes4cgp::set_verbosity)
        .def("get_name", &dcgp::moes4cgp::get_name)
        .def("get_extra_info", &dcgp::moes4cgp::get_extra_info)
        .def("get_seed", &dcgp::moes4cgp::get_seed, generic_uda_get_seed_doc().c_str())
        .def("set_bfe", &dcgp::moes4cgp::set_bfe, generic_set_bfe_doc().c_str(), py::arg("b"))
        .def("get_log", &generic_log_getter<dcgp::moes4cgp>, moes4cgp_get_log_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::moes4cgp>, &udx_pickle_setstate<dcgp::moes4cgp>))
        .def("__repr__", &dcgp::moes4cgp::get_extra_info);
    // GD-4CGP (Gradient Descent for Cartesian Genetic Programming)
    py::class_<dcgp::gd4cgp> gd4cgp_(m, "gd4cgp", gd4cgp_doc().c_str());
    gd4cgp_
        .def(py::init<unsigned, double, double>(), py::arg("max_iter") = 1u, py::arg("lr") = 1., py::arg("lr_min") = 0.)
        .def("evolve", &dcgp::gd4cgp::evolve)
        .def("set_verbosity", &dcgp::gd4cgp::set_verbosity)
        .def("get_name", &dcgp::gd4cgp::get_name)
        .def("get_extra_info", &dcgp::gd4cgp::get_extra_info)
        .def("get_log", &generic_log_getter<dcgp::gd4cgp>, gd4cgp_get_log_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::gd4cgp>, &udx_pickle_setstate<dcgp::gd4cgp>))
        .def("__repr__", &dcgp::gd4cgp::get_extra_info);
    // MES-4CGP (Memetic Evolutionary Strategy for Cartesian Genetic Programming)
    py::class_<dcgp::mes4cgp> mes4cgp_(m, "mes4cgp", mes4cgp_doc().c_str());
    mes4cgp_
        .def(py::init<unsigned, unsigned, double>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u, py::arg("ftol") = 0.)
        .def(py::init<unsigned, unsigned, double, unsigned>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("seed"))
        .def("evolve", &dcgp::mes4cgp::evolve)
        .def("set_verbosity", &dcgp::mes4cgp::set_verbosity)
        .def("get_name", &dcgp::mes4cgp::get_name)
        .def("get_extra_info", &dcgp::mes4cgp::get_extra_info)
        .def("get_seed", &dcgp::mes4cgp::get_seed, generic_uda_get_seed_doc().c_str())
        .def("get_log", &generic_log_getter<dcgp::mes4cgp>, mes4cgp_get_log_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::mes4cgp>, &udx_pickle_setstate<dcgp::mes4cgp>))
        .def("__repr__", &dcgp::mes4cgp::get_extra_info);
    // MOMES-4CGP (Multi-Objective Memetic Evolutionary Strategy for Cartesian Genetic Programming)
    py::class_<dcgp::momes4cgp> momes4cgp_(m, "momes4cgp", momes4cgp_doc().c_str());

    momes4cgp_
        .def(py::init<unsigned, unsigned, double>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0.)
        .def(py::init<unsigned, unsigned, double, unsigned>(), py::arg("gen") = 1u, py::arg("max_mut") = 4u,
             py::arg("ftol") = 0., py::arg("seed"))
        .def("evolve", &dcgp::momes4cgp::evolve)
        .def("set_verbosity", &dcgp::momes4cgp::set_verbosity)
        .def("get_name", &dcgp::momes4cgp::get_name)
        .def("get_extra_info", &dcgp::momes4cgp::get_extra_info)
        .def("get_seed", &dcgp::momes4cgp::get_seed, generic_uda_get_seed_doc().c_str())
        .def("get_log", &generic_log_getter<dcgp::momes4cgp>, momes4cgp_get_log_doc().c_str())
        .def(py::pickle(&udx_pickle_getstate<dcgp::momes4cgp>, &udx_pickle_setstate<dcgp::momes4cgp>))
        .def("__repr__", &dcgp::momes4cgp::get_extra_info);

    // Making data from the gym available in python
    expose_data_from_the_gym<&gym::generate_koza_quintic>(m, "generate_koza_quintic", generate_koza_quintic_doc());
    // From Our paper
    expose_data_from_the_gym<&gym::generate_P1>(m, "generate_P1", generate_P1_doc());
    expose_data_from_the_gym<&gym::generate_P2>(m, "generate_P2", generate_P2_doc());
    expose_data_from_the_gym<&gym::generate_P3>(m, "generate_P3", generate_P3_doc());
    expose_data_from_the_gym<&gym::generate_P4>(m, "generate_P4", generate_P4_doc());
    expose_data_from_the_gym<&gym::generate_P5>(m, "generate_P5", generate_P5_doc());
    expose_data_from_the_gym<&gym::generate_P6>(m, "generate_P6", generate_P6_doc());
    expose_data_from_the_gym<&gym::generate_P7>(m, "generate_P7", generate_P7_doc());
    // From Vladi paper
    expose_data_from_the_gym<&gym::generate_kotanchek>(m, "generate_kotanchek", generate_kotanchek_doc());
    expose_data_from_the_gym<&gym::generate_salutowicz>(m, "generate_salutowicz", generate_salutowicz_doc());
    expose_data_from_the_gym<&gym::generate_salutowicz2d>(m, "generate_salutowicz2d", generate_salutowicz2d_doc());
    expose_data_from_the_gym<&gym::generate_uball5d>(m, "generate_uball5d", generate_uball5d_doc());
    expose_data_from_the_gym<&gym::generate_ratpol3d>(m, "generate_ratpol3d", generate_ratpol3d_doc());
    expose_data_from_the_gym<&gym::generate_sinecosine>(m, "generate_sinecosine", generate_sinecosine_doc());
    expose_data_from_the_gym<&gym::generate_ripple>(m, "generate_ripple", generate_ripple_doc());
    expose_data_from_the_gym<&gym::generate_ratpol2d>(m, "generate_ratpol2d", generate_ratpol2d_doc());
    // NIST data
    expose_data_from_the_gym<&gym::generate_chwirut1>(m, "generate_chwirut1", generate_chwirut1_doc());
    expose_data_from_the_gym<&gym::generate_chwirut2>(m, "generate_chwirut2", generate_chwirut2_doc());
    expose_data_from_the_gym<&gym::generate_daniel_wood>(m, "generate_daniel_wood", generate_daniel_wood_doc());
    expose_data_from_the_gym<&gym::generate_gauss1>(m, "generate_gauss1", generate_gauss1_doc());
    expose_data_from_the_gym<&gym::generate_kirby2>(m, "generate_kirby2", generate_kirby2_doc());
    expose_data_from_the_gym<&gym::generate_lanczos2>(m, "generate_lanczos2", generate_lanczos2_doc());
    expose_data_from_the_gym<&gym::generate_misra1b>(m, "generate_misra1b", generate_misra1b_doc());
    // MISC data
    expose_data_from_the_gym<&gym::generate_luca1>(m, "generate_luca1", generate_luca1_doc());

}
} // namespace dcgpy
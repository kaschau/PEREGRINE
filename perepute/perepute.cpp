#include "user.hpp"

//#if defined(__GNUC__)
//#  pragma GCC diagnostic push
//#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
//#endif

#include <pybind11/pybind11.h>

//#if defined(__GNUC__)
//#  pragma GCC diagnostic pop
//#endif

#include <cstdlib>

namespace py = pybind11;

//--------------------------------------------------------------------------------------//
//
//        The python module
//
//--------------------------------------------------------------------------------------//

PYBIND11_MODULE(perepute, m) {
  ///
  /// This is a python binding to the user-defined generate_view function
  /// declared in user.hpp which returns a Kokkos::View. This function is called
  /// from ex-numpy.py
  ///
  m.def("generate_view", &generate_view, "Generate a random view",
         py::arg("n") = 10);

  m.def("generate_view2", &generate_view2, "Generate a random view",
         py::arg("n") = 15);

  static auto _atexit = []() {
    if (Kokkos::is_initialized()) Kokkos::finalize();
  };

  atexit(_atexit);
}


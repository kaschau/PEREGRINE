#include <kokkosTypes.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string.h>

namespace py = pybind11;

void bindKokkos(py::module_ &m) {
  // ./utils
  py::module_ pgkokkos = m.def_submodule("pgkokkos", "pgkokkos module");

  // ONE D VIEW
  py::class_<oneDview>(pgkokkos, "view1").def(py::init<std::string, size_t>());

  py::class_<oneDview::host_mirror_type>(pgkokkos, "mirror1",
                                         py::buffer_protocol())
      .def(py::init([](oneDview &view) {
        oneDview::host_mirror_type *mirror = new oneDview::host_mirror_type();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](oneDview::host_mirror_type &view) -> py::buffer_info {
        size_t shape[1] = {view.extent(0)};
        size_t stride[1] = {sizeof(double) * view.stride(0)};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            1,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  // TWO D VIEW
  py::class_<twoDview>(pgkokkos, "view2")
      .def(py::init<std::string, size_t, size_t>());

  py::class_<twoDview::host_mirror_type>(pgkokkos, "mirror2",
                                         py::buffer_protocol())
      .def(py::init([](twoDview &view) {
        twoDview::host_mirror_type *mirror = new twoDview::host_mirror_type();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](twoDview::host_mirror_type &view) -> py::buffer_info {
        size_t shape[2] = {view.extent(0), view.extent(1)};
        size_t stride[2] = {sizeof(double) * view.stride(0),
                            sizeof(double) * view.stride(1)};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            2,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  // THREE D VIEW
  py::class_<threeDview>(pgkokkos, "view3")
      .def(py::init<std::string, size_t, size_t, size_t>());

  py::class_<threeDview::host_mirror_type>(pgkokkos, "mirror3",
                                           py::buffer_protocol())
      .def(py::init([](threeDview &view) {
        threeDview::host_mirror_type *mirror =
            new threeDview::host_mirror_type();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](threeDview::host_mirror_type &view) -> py::buffer_info {
        size_t shape[3] = {view.extent(0), view.extent(1), view.extent(2)};
        size_t stride[3] = {sizeof(double) * view.stride(0),
                            sizeof(double) * view.stride(1),
                            sizeof(double) * view.stride(2)};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            3,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  // FOUR D VIEW
  py::class_<fourDview>(pgkokkos, "view4")
      .def(py::init<std::string, size_t, size_t, size_t, size_t>());

  py::class_<fourDview::host_mirror_type>(pgkokkos, "mirror4",
                                          py::buffer_protocol())
      .def(py::init([](fourDview &view) {
        fourDview::host_mirror_type *mirror = new fourDview::host_mirror_type();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](fourDview::host_mirror_type &view) -> py::buffer_info {
        size_t shape[4] = {view.extent(0), view.extent(1), view.extent(2),
                           view.extent(3)};
        size_t stride[4] = {
            sizeof(double) * view.stride(0), sizeof(double) * view.stride(1),
            sizeof(double) * view.stride(2), sizeof(double) * view.stride(3)};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            4,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  // FIVE D VIEW
  py::class_<fiveDview>(pgkokkos, "view5")
      .def(py::init<std::string, size_t, size_t, size_t, size_t, size_t>());

  py::class_<fiveDview::host_mirror_type>(pgkokkos, "mirror5",
                                          py::buffer_protocol())
      .def(py::init([](fiveDview &view) {
        fiveDview::host_mirror_type *mirror = new fiveDview::host_mirror_type();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](fiveDview::host_mirror_type &view) -> py::buffer_info {
        size_t shape[5] = {view.extent(0), view.extent(1), view.extent(2),
                           view.extent(3), view.extent(4)};
        size_t stride[5] = {
            sizeof(double) * view.stride(0), sizeof(double) * view.stride(1),
            sizeof(double) * view.stride(2), sizeof(double) * view.stride(3),
            sizeof(double) * view.stride(4)};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            5,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  pgkokkos.def("initialize", []() { Kokkos::initialize(); });
  pgkokkos.def("finalize", []() { Kokkos::finalize(); });

  pgkokkos.def(
      "deep_copy",
      [](oneDview &dest, oneDview::host_mirror_type &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_oneHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](oneDview::host_mirror_type &dest, oneDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_oneDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](twoDview &dest, twoDview::host_mirror_type &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_twoHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](twoDview::host_mirror_type &dest, twoDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_twoDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](threeDview &dest, threeDview::host_mirror_type &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_threeHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](threeDview::host_mirror_type &dest, threeDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_threeDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](fourDview &dest, fourDview::host_mirror_type &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fourHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](fourDview::host_mirror_type &dest, fourDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fourDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](fiveDview &dest, fiveDview::host_mirror_type &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fiveHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](fiveDview::host_mirror_type &dest, fiveDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fiveDH", py::arg("dest"), py::arg("src"));
}

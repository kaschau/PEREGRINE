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

  py::class_<oneDview::HostMirror>(pgkokkos, "mirror1", py::buffer_protocol())
      .def(py::init([](oneDview &view) {
        oneDview::HostMirror *mirror = new oneDview::HostMirror();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](oneDview::HostMirror &view) -> py::buffer_info {
        size_t shape[1] = {view.extent(0)};
        size_t stride[1] = {sizeof(double) * view.stride_0()};
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

  py::class_<twoDview::HostMirror>(pgkokkos, "mirror2", py::buffer_protocol())
      .def(py::init([](twoDview &view) {
        twoDview::HostMirror *mirror = new twoDview::HostMirror();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](twoDview::HostMirror &view) -> py::buffer_info {
        size_t shape[2] = {view.extent(0), view.extent(1)};
        size_t stride[2] = {sizeof(double) * view.stride_0(),
                            sizeof(double) * view.stride_1()};
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

  py::class_<threeDview::HostMirror>(pgkokkos, "mirror3", py::buffer_protocol())
      .def(py::init([](threeDview &view) {
        threeDview::HostMirror *mirror = new threeDview::HostMirror();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](threeDview::HostMirror &view) -> py::buffer_info {
        size_t shape[3] = {view.extent(0), view.extent(1), view.extent(2)};
        size_t stride[3] = {sizeof(double) * view.stride_0(),
                            sizeof(double) * view.stride_1(),
                            sizeof(double) * view.stride_2()};
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

  py::class_<fourDview::HostMirror>(pgkokkos, "mirror4", py::buffer_protocol())
      .def(py::init([](fourDview &view) {
        fourDview::HostMirror *mirror = new fourDview::HostMirror();
        *mirror = Kokkos::create_mirror_view(view);
        return mirror;
      }))
      .def_buffer([](fourDview::HostMirror &view) -> py::buffer_info {
        size_t shape[4] = {view.extent(0), view.extent(1), view.extent(2),
                           view.extent(3)};
        size_t stride[4] = {
            sizeof(double) * view.stride_0(), sizeof(double) * view.stride_1(),
            sizeof(double) * view.stride_2(), sizeof(double) * view.stride_3()};
        return py::buffer_info(
            view.data(),                             // Pointer to buffer
            sizeof(double),                          // Size of one scalar
            py::format_descriptor<double>::format(), // Descriptor
            4,                                       // Number of dimensions
            shape,                                   // Buffer dimensions
            stride // Strides (in bytes) for each index
        );
      });

  // FIVE D HOST VIEW
  py::class_<fiveDviewHost>(pgkokkos, "hostView5", py::buffer_protocol())
      .def(py::init<std::string, size_t, size_t, size_t, size_t, size_t>())
      .def_buffer([](fiveDviewHost &view) -> py::buffer_info {
        size_t shape[5] = {view.extent(0), view.extent(1), view.extent(2),
                           view.extent(3), view.extent(4)};
        size_t stride[5] = {
            sizeof(double) * view.stride_0(), sizeof(double) * view.stride_1(),
            sizeof(double) * view.stride_2(), sizeof(double) * view.stride_3(),
            sizeof(double) * view.stride_4()};
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
      [](oneDview &dest, oneDview::HostMirror &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_oneHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](oneDview::HostMirror &dest, oneDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_oneDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](twoDview &dest, twoDview::HostMirror &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_twoHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](twoDview::HostMirror &dest, twoDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_twoDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](threeDview &dest, threeDview::HostMirror &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_threeHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](threeDview::HostMirror &dest, threeDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_threeDH", py::arg("dest"), py::arg("src"));

  pgkokkos.def(
      "deep_copy",
      [](fourDview &dest, fourDview::HostMirror &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fourHD", py::arg("dest"), py::arg("src"));
  pgkokkos.def(
      "deep_copy",
      [](fourDview::HostMirror &dest, fourDview &src) {
        Kokkos::deep_copy(dest, src);
      },
      "deep_copy_fourDH", py::arg("dest"), py::arg("src"));
}

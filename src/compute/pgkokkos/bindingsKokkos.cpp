#include <kokkosTypes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void initialize() { Kokkos::initialize(); }

void finalize() { Kokkos::finalize(); }

void deep_copy(oneDview &dest, oneDview::HostMirror &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(twoDview &dest, twoDview::HostMirror &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(threeDview &dest, threeDview::HostMirror &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(fourDview &dest, fourDview::HostMirror &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(oneDview::HostMirror &dest, oneDview &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(twoDview::HostMirror &dest, twoDview &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(threeDview::HostMirror &dest, threeDview &src) {
  Kokkos::deep_copy(dest, src);
}

void deep_copy(fourDview::HostMirror &dest, fourDview &src) {
  Kokkos::deep_copy(dest, src);
}

void bindKokkos(nb::module_ &m) {
  // ./utils
  nb::module_ pgkokkos = m.def_submodule("pgkokkos", "pgkokkos module");

  // ONE D VIEW
  nb::class_<oneDview>(pgkokkos, "array1").def(nb::init<std::string, size_t>());

  nb::class_<oneDview::HostMirror>(pgkokkos, "mirror1")
      .def("__init__",
           [](oneDview::HostMirror *mirror, oneDview &view) {
             new (mirror) oneDview::HostMirror;
             *mirror = create_mirror_view(view);
           })
      .def("__array__",
           [](oneDview::HostMirror *view)
               -> nb::ndarray<nb::numpy, double, nb::shape<nb::any>> {
             size_t shape[1] = {view->extent(0)};
             int64_t stride[1] = {static_cast<int64_t>(view->stride_0())};
             return nb::ndarray<nb::numpy, double, nb::shape<nb::any>>(
                 view->data(), 1, shape, nb::handle(), stride);
           });

  // TWO D VIEW
  nb::class_<twoDview>(pgkokkos, "array2")
      .def(nb::init<std::string, size_t, size_t>());

  nb::class_<twoDview::HostMirror>(pgkokkos, "mirror2")
      .def("__init__",
           [](twoDview::HostMirror *mirror, twoDview &view) {
             new (mirror) twoDview::HostMirror;
             *mirror = create_mirror_view(view);
           })
      .def("__array__",
           [](twoDview::HostMirror *view)
               -> nb::ndarray<nb::numpy, double, nb::shape<nb::any, nb::any>> {
             size_t shape[2] = {view->extent(0), view->extent(0)};
             int64_t stride[2] = {static_cast<int64_t>(view->stride_0()),
                                  static_cast<int64_t>(view->stride_1())};
             return nb::ndarray<nb::numpy, double, nb::shape<nb::any, nb::any>>(
                 view->data(), 2, shape, nb::handle(), stride);
           });

  // THREE D VIEW
  nb::class_<threeDview>(pgkokkos, "array3")
      .def(nb::init<std::string, size_t, size_t, size_t>());

  nb::class_<threeDview::HostMirror>(pgkokkos, "mirror3")
      .def("__init__",
           [](threeDview::HostMirror *mirror, threeDview &view) {
             new (mirror) threeDview::HostMirror;
             *mirror = create_mirror_view(view);
           })
      .def("__array__",
           [](threeDview::HostMirror *view)
               -> nb::ndarray<nb::numpy, double,
                              nb::shape<nb::any, nb::any, nb::any>> {
             size_t shape[3] = {view->extent(0), view->extent(1),
                                view->extent(2)};
             int64_t stride[3] = {static_cast<int64_t>(view->stride_0()),
                                  static_cast<int64_t>(view->stride_1()),
                                  static_cast<int64_t>(view->stride_2())};
             return nb::ndarray<nb::numpy, double,
                                nb::shape<nb::any, nb::any, nb::any>>(
                 view->data(), 3, shape, nb::handle(), stride);
           });

  // FOUR D VIEW
  nb::class_<fourDview>(pgkokkos, "array4")
      .def(nb::init<std::string, size_t, size_t, size_t, size_t>());

  nb::class_<fourDview::HostMirror>(pgkokkos, "mirror4")
      .def("__init__",
           [](fourDview::HostMirror *mirror, fourDview &view) {
             new (mirror) fourDview::HostMirror;
             *mirror = create_mirror_view(view);
           })
      .def("__array__",
           [](fourDview::HostMirror *view)
               -> nb::ndarray<nb::numpy, double,
                              nb::shape<nb::any, nb::any, nb::any, nb::any>> {
             size_t shape[4] = {view->extent(0), view->extent(1),
                                view->extent(2), view->extent(3)};
             int64_t stride[4] = {static_cast<int64_t>(view->stride_0()),
                                  static_cast<int64_t>(view->stride_1()),
                                  static_cast<int64_t>(view->stride_2()),
                                  static_cast<int64_t>(view->stride_3())};
             return nb::ndarray<nb::numpy, double,
                                nb::shape<nb::any, nb::any, nb::any, nb::any>>(
                 view->data(), 4, shape, nb::handle(), stride);
           });

  pgkokkos.def("initialize", &initialize);
  pgkokkos.def("finalize", &finalize);
  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<oneDview &, oneDview::HostMirror &>(&deep_copy),
      "deep_copy_one", nb::arg("dest"), nb::arg("src"));
  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<oneDview::HostMirror &, oneDview &>(&deep_copy),
      "deep_copy_one", nb::arg("dest"), nb::arg("src"));

  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<twoDview &, twoDview::HostMirror &>(&deep_copy),
      "deep_copy_two", nb::arg("dest"), nb::arg("src"));
  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<twoDview::HostMirror &, twoDview &>(&deep_copy),
      "deep_copy_two", nb::arg("dest"), nb::arg("src"));

  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<threeDview &, threeDview::HostMirror &>(&deep_copy),
      "deep_copy_three", nb::arg("dest"), nb::arg("src"));
  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<threeDview::HostMirror &, threeDview &>(&deep_copy),
      "deep_copy_three", nb::arg("dest"), nb::arg("src"));

  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<fourDview &, fourDview::HostMirror &>(&deep_copy),
      "deep_copy_four", nb::arg("dest"), nb::arg("src"));
  pgkokkos.def(
      "deep_copy",
      nb::overload_cast<fourDview::HostMirror &, fourDview &>(&deep_copy),
      "deep_copy_four", nb::arg("dest"), nb::arg("src"));
}

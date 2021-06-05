#include "user.hpp"
#include "Kokkos_Core.hpp"
#include <cstdint>
#include <iostream>

//struct InitView {
//  explicit InitView(view_type _v) : m_view(_v) {}
//
//  KOKKOS_INLINE_FUNCTION
//  void operator()(const int i) const { m_view(i, i % 2) = i; }
//
// private:
//  view_type m_view;
//};

///
/// \fn generate_view
/// \brief This is meant to emulate some function that exists in a user library
/// which returns a Kokkos::View and will have a python binding
///
//view_type generate_view(size_t n) {
  //if (!Kokkos::is_initialized()) {
    //std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    //Kokkos::initialize();
  //}
  //view_type _v("user_view", n, 2);
  //Kokkos::RangePolicy<exec_space, int> range(0, n);
  //Kokkos::parallel_for("generate_view", range, InitView{_v});
  //return _v;
//}

void add(oneDview kkview, double n) {

  auto len = kkview.size();
  std::cout << kkview.size() << std::endl;
  Kokkos::parallel_for("add", len, KOKKOS_LAMBDA(const int i) {
    kkview(i) += n;
  });
}

void add2(threeDview kkview, double n,
          int imin, int jmin, int kmin,
          int imax, int jmax, int kmax) {

  MDRange3 _range({{imin,jmin,kmin}},{{imax,jmax,kmax}});
  Kokkos::parallel_for("add2", _range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
      kkview(i,j,k) += n - 1.0;
      kkview(i,j,k) += n + 1.0*2.0;
      kkview(i,j,k) += n / 1.0;
      kkview(i,j,k) += n - 1.0;
      kkview(i,j,k) += n;
      kkview(i,j,k) += n;
      kkview(i,j,k) += n;
      kkview(i,j,k) += n;
      kkview(i,j,k) += n;
  });
}

void add3(block b, double n ) {

  MDRange3 _range({{0,0,0}},{{b.nx,b.ny,b.nz}});
  Kokkos::parallel_for("add3", _range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
      std::cout << b.nblki << "\n";

      b.x(i,j,k) += n;
  });
}


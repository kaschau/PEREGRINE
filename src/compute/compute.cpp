
#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "Block.hpp"

void add3D(Block b, double n ) {

  MDRange3 _range({{0,0,0}},{{b.ni,b.nj,b.nk}});
  Kokkos::parallel_for("add3", _range, KOKKOS_LAMBDA(const int i, const int j, const int k) {

      b.x(i,j,k) += n;
      b.y(i,j,k) += n;
      b.z(i,j,k) += n;

  });
}


threeDview gen3Dview(std::string name, int ni, int nj, int nk) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  threeDview _v(name, ni, nj, nk);
  MDRange3 _range({{0,0,0}},{{ni,nj,nk}});
  Kokkos::parallel_for("Gen3", _range, KOKKOS_LAMBDA(const int i,
                                                     const int j,
                                                     const int k) {
    _v(i,j,k) = 0.0;
  });
  return _v;
}

fourDview gen4Dview(std::string name, int ni, int nj, int nk, int nl) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  fourDview _v(name, ni, nj, nk, nl);
  MDRange4 _range({{0,0,0,0}},{{ni,nj,nk,nl}});
  Kokkos::parallel_for("Gen4", _range, KOKKOS_LAMBDA(const int i,
                                                     const int j,
                                                     const int k,
                                                     const int l) {
    _v(i,j,k,l) = 0.0;
  });
  return _v;
}

void finalize_kokkos() {
  if (Kokkos::is_initialized()) {
    std::cerr << "Finalizing Kokkos..." << std::endl;
    Kokkos::finalize();
  }
}

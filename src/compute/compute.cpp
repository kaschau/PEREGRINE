
#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void add3D(block_ b, double n ) {

  MDRange3 range({{0,0,0}},{{b.ni,b.nj,b.nk}});
  Kokkos::parallel_for("add3", range, KOKKOS_LAMBDA(const int i, const int j, const int k) {

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
  threeDview v(name, ni, nj, nk);
  MDRange3 range({{0,0,0}},{{ni,nj,nk}});
  Kokkos::parallel_for("Gen3", range, KOKKOS_LAMBDA(const int i,
                                                    const int j,
                                                    const int k) {

    v(i,j,k) = 0.0;

  });
  return v;
}

fourDview gen4Dview(std::string name, int ni, int nj, int nk, int nl) {
  if (!Kokkos::is_initialized()) {
    std::cerr << "[user-bindings]> Initializing Kokkos..." << std::endl;
    Kokkos::initialize();
  }
  fourDview v(name, ni, nj, nk, nl);
  MDRange4 range({{0,0,0,0}},{{ni,nj,nk,nl}});
  Kokkos::parallel_for("Gen4", range, KOKKOS_LAMBDA(const int i,
                                                    const int j,
                                                    const int k,
                                                    const int l) {

    v(i,j,k,l) = 0.0;

  });
  return v;
}

void finalize_kokkos() {
  if (Kokkos::is_initialized()) {
    std::cerr << "Finalizing Kokkos..." << std::endl;
    Kokkos::finalize();
  }
}

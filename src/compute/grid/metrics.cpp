#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>
#include <math.h>

void metrics(std::vector<block_> mb) {
for(block_ b : mb){

  // Cell center range
  MDRange3 range_c({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_c, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
//----------------------------------------------------------------------------//
//Cell Centers
//----------------------------------------------------------------------------//
  b.xc(i,j,k) = 0.125 * ( b.x(i  ,j  ,k) + b.x(i+1,j,k  ) + b.x(i  ,j+1,k  ) + b.x(i,j  ,k+1)
                        + b.x(i+1,j+1,k) + b.x(i+1,j,k+1) + b.x(i+1,j+1,k+1) + b.x(i,j+1,k+1) );

  b.yc(i,j,k) = 0.125 * ( b.y(i  ,j  ,k) + b.y(i+1,j,k  ) + b.y(i  ,j+1,k  ) + b.y(i,j  ,k+1)
                        + b.y(i+1,j+1,k) + b.y(i+1,j,k+1) + b.y(i+1,j+1,k+1) + b.y(i,j+1,k+1) );

  b.zc(i,j,k) = 0.125 * ( b.z(i  ,j  ,k) + b.z(i+1,j,k  ) + b.z(i  ,j+1,k  ) + b.z(i,j  ,k+1)
                        + b.z(i+1,j+1,k) + b.z(i+1,j,k+1) + b.z(i+1,j+1,k+1) + b.z(i,j+1,k+1) );
  });

// i face range
  MDRange3 range_i({0,0,0},{b.ni+2,b.nj+1,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_i, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
//----------------------------------------------------------------------------//
// i face area vector
//----------------------------------------------------------------------------//
  b.isx(i,j,k) = 0.5 * ( (b.y(i,j+1,k+1)-b.y(i,j  ,k)) *
                         (b.z(i,j  ,k+1)-b.z(i,j+1,k)) -
                         (b.z(i,j+1,k+1)-b.z(i,j  ,k)) *
                         (b.y(i,j  ,k+1)-b.y(i,j+1,k)) );

  b.isy(i,j,k) = 0.5 * ( (b.z(i,j+1,k+1)-b.z(i,j  ,k)) *
                         (b.x(i,j  ,k+1)-b.x(i,j+1,k)) -
                         (b.x(i,j+1,k+1)-b.x(i,j  ,k)) *
                         (b.z(i,j  ,k+1)-b.z(i,j+1,k)) );


  b.isz(i,j,k) = 0.5 * ( (b.x(i,j+1,k+1)-b.x(i,j  ,k)) *
                         (b.y(i,j  ,k+1)-b.y(i,j+1,k)) -
                         (b.y(i,j+1,k+1)-b.y(i,j  ,k)) *
                         (b.x(i,j  ,k+1)-b.x(i,j+1,k)) );


  b.iS (i,j,k) =   sqrt( pow(b.isx(i,j,k),2.0) +
                         pow(b.isy(i,j,k),2.0) +
                         pow(b.isz(i,j,k),2.0) );

  b.inx(i,j,k) = b.isx(i,j,k)/b.iS(i,j,k);
  b.iny(i,j,k) = b.isy(i,j,k)/b.iS(i,j,k);
  b.inz(i,j,k) = b.isz(i,j,k)/b.iS(i,j,k);

  });


  // j face range
  MDRange3 range_j({0,0,0},{b.ni+1,b.nj+2,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_j, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
//----------------------------------------------------------------------------//
// j face area vector
//----------------------------------------------------------------------------//
  b.jsx(i,j,k) = 0.5 * ( (b.y(i+1,j,k+1)-b.y(i,j,k  )) *
                         (b.z(i+1,j,k  )-b.z(i,j,k+1)) -
                         (b.z(i+1,j,k+1)-b.z(i,j,k  )) *
                         (b.y(i+1,j,k  )-b.y(i,j,k+1)) );

  b.jsy(i,j,k) = 0.5 * ( (b.z(i+1,j,k+1)-b.z(i,j,k  )) *
                         (b.x(i+1,j,k  )-b.x(i,j,k+1)) -
                         (b.x(i+1,j,k+1)-b.x(i,j,k  )) *
                         (b.z(i+1,j,k  )-b.z(i,j,k+1)) );


  b.jsz(i,j,k) = 0.5 * ( (b.x(i+1,j,k+1)-b.x(i,j,k  )) *
                         (b.y(i+1,j,k  )-b.y(i,j,k+1)) -
                         (b.y(i+1,j,k+1)-b.y(i,j,k  )) *
                         (b.x(i+1,j,k  )-b.x(i,j,k+1)) );


  b.jS (i,j,k) =   sqrt( pow(b.jsx(i,j,k),2.0) +
                         pow(b.jsy(i,j,k),2.0) +
                         pow(b.jsz(i,j,k),2.0) );

  b.jnx(i,j,k) = b.jsx(i,j,k)/b.jS(i,j,k);
  b.jny(i,j,k) = b.jsy(i,j,k)/b.jS(i,j,k);
  b.jnz(i,j,k) = b.jsz(i,j,k)/b.jS(i,j,k);

  });

  // k face range
  MDRange3 range_k({0,0,0},{b.ni+1,b.nj+1,b.nk+2});
  Kokkos::parallel_for("Grid Metrics", range_k, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
//----------------------------------------------------------------------------//
// k face area vector
//----------------------------------------------------------------------------//
  b.ksx(i,j,k) = 0.5 * ( (b.y(i+1,j+1,k)-b.y(i  ,j,k)) *
                         (b.z(i  ,j+1,k)-b.z(i+1,j,k)) -
                         (b.z(i+1,j+1,k)-b.z(i  ,j,k)) *
                         (b.y(i  ,j+1,k)-b.y(i+1,j,k)) );

  b.ksy(i,j,k) = 0.5 * ( (b.z(i+1,j+1,k)-b.z(i  ,j,k)) *
                         (b.x(i  ,j+1,k)-b.x(i+1,j,k)) -
                         (b.x(i+1,j+1,k)-b.x(i  ,j,k)) *
                         (b.z(i  ,j+1,k)-b.z(i+1,j,k)) );


  b.ksz(i,j,k) = 0.5 * ( (b.x(i+1,j+1,k)-b.x(i  ,j,k)) *
                         (b.y(i  ,j+1,k)-b.y(i+1,j,k)) -
                         (b.y(i+1,j+1,k)-b.y(i  ,j,k)) *
                         (b.x(i  ,j+1,k)-b.x(i+1,j,k)) );


  b.kS (i,j,k) =   sqrt( pow(b.ksx(i,j,k),2.0) +
                         pow(b.ksy(i,j,k),2.0) +
                         pow(b.ksz(i,j,k),2.0) );

  b.knx(i,j,k) = b.ksx(i,j,k)/b.kS(i,j,k);
  b.kny(i,j,k) = b.ksy(i,j,k)/b.kS(i,j,k);
  b.knz(i,j,k) = b.ksz(i,j,k)/b.kS(i,j,k);

  });

}};

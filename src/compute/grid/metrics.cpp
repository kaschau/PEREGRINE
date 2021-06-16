#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <iostream>
#include <math.h>

void metrics(block_ b) {

  MDRange3 range_c({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_c, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
  // Cell Centers
  b.xc(i,j,k) = 0.125 * ( b.x(i  ,j  ,k) + b.x(i+1,j,k  ) + b.x(i  ,j+1,k  ) + b.x(i,j  ,k+1)
                        + b.x(i+1,j+1,k) + b.x(i+1,j,k+1) + b.x(i+1,j+1,k+1) + b.x(i,j+1,k+1) );

  b.yc(i,j,k) = 0.125 * ( b.y(i  ,j  ,k) + b.y(i+1,j,k  ) + b.y(i  ,j+1,k  ) + b.y(i,j  ,k+1)
                        + b.y(i+1,j+1,k) + b.y(i+1,j,k+1) + b.y(i+1,j+1,k+1) + b.y(i,j+1,k+1) );

  b.zc(i,j,k) = 0.125 * ( b.z(i  ,j  ,k) + b.z(i+1,j,k  ) + b.z(i  ,j+1,k  ) + b.z(i,j  ,k+1)
                        + b.z(i+1,j+1,k) + b.z(i+1,j,k+1) + b.z(i+1,j+1,k+1) + b.z(i,j+1,k+1) );
  });

  MDRange3 range_i({0,0,0},{b.ni+2,b.nj+1,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_i, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
  // I face area vector
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

  });

}

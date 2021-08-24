#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>
#include <math.h>

void metrics(std::vector<block_> mb) {
for(block_ b : mb){

  // Cell center range
  MDRange3 range_c({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
  Kokkos::parallel_for("xc Grid Metrics", range_c, KOKKOS_LAMBDA(const int i,
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
  Kokkos::parallel_for("is Grid Metrics", range_i, KOKKOS_LAMBDA(const int i,
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
  Kokkos::parallel_for("js Grid Metrics", range_j, KOKKOS_LAMBDA(const int i,
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
  Kokkos::parallel_for("ks Grid Metrics", range_k, KOKKOS_LAMBDA(const int i,
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

  // Cell center range
  MDRange3 range_J({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
  Kokkos::parallel_for("J Grid Metrics", range_J, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
//----------------------------------------------------------------------------//
//Cell Volumes
//----------------------------------------------------------------------------//
  b.J(i,j,k) = ( ( b.x(i+1,j+1,k+1) - b.x(i,j,k) )*( b.isx(i+1,j  ,k  )
                                                   + b.jsx(i  ,j+1,k  )
                                                   + b.ksx(i  ,j  ,k+1) )
               + ( b.y(i+1,j+1,k+1) - b.y(i,j,k) )*( b.isy(i+1,j  ,k  )
                                                   + b.jsy(i  ,j+1,k  )
                                                   + b.ksy(i  ,j  ,k+1) )
               + ( b.z(i+1,j+1,k+1) - b.z(i,j,k) )*( b.isz(i+1,j  ,k  )
                                                   + b.jsz(i  ,j+1,k  )
                                                   + b.ksz(i  ,j  ,k+1) ) )/3.e0;

//----------------------------------------------------------------------------//
//Cell center transformation metrics (for FD diffusion operator)
//----------------------------------------------------------------------------//

  double x1,x2,x3,x4,x5,x6,x7,x8;
  double y1,y2,y3,y4,y5,y6,y7,y8;
  double z1,z2,z3,z4,z5,z6,z7,z8;

  double dxdE, dydE, dzdE;
  double dxdN, dydN, dzdN;
  double dxdX, dydX, dzdX;

  // Cell corners
  x1 = b.x(i,  j,  k  ); x2 = b.x(i  ,j+1,  k); x3 = b.x(i+1,j+1,k  ); x4 = b.x(i+1,j  ,k  );
  x5 = b.x(i  ,j  ,k+1); x6 = b.x(i  ,j+1,k+1); x7 = b.x(i+1,j+1,k+1); x8 = b.x(i+1,j  ,k+1);

  y1 = b.y(i,  j,  k  ); y2 = b.y(i  ,j+1,  k); y3 = b.y(i+1,j+1,k  ); y4 = b.y(i+1,j  ,k  );
  y5 = b.y(i  ,j  ,k+1); y6 = b.y(i  ,j+1,k+1); y7 = b.y(i+1,j+1,k+1); y8 = b.y(i+1,j  ,k+1);

  z1 = b.z(i,  j,  k  ); z2 = b.z(i  ,j+1,  k); z3 = b.z(i+1,j+1,k  ); z4 = b.z(i+1,j  ,k  );
  z5 = b.z(i  ,j  ,k+1); z6 = b.z(i  ,j+1,k+1); z7 = b.z(i+1,j+1,k+1); z8 = b.z(i+1,j  ,k+1);

  //Derivative of (x,y,z) w.r.t. (E,N,X)
  dxdE = 0.25*( (x4-x1) + (x8-x5) + (x3-x2) + (x7-x6) );
  dydE = 0.25*( (y4-y1) + (y8-y5) + (y3-y2) + (y7-y6) );
  dzdE = 0.25*( (z4-z1) + (z8-z5) + (z3-z2) + (z7-z6) );

  dxdN = 0.25*( (x2-x1) + (x3-x4) + (x7-x8) + (x6-x5) );
  dydN = 0.25*( (y2-y1) + (y3-y4) + (y7-y8) + (y6-y5) );
  dzdN = 0.25*( (z2-z1) + (z3-z4) + (z7-z8) + (z6-z5) );

  dxdX = 0.25*( (x5-x1) + (x8-x4) + (x6-x2) + (x7-x3) );
  dydX = 0.25*( (y5-y1) + (y8-y4) + (y6-y2) + (y7-y3) );
  dzdX = 0.25*( (z5-z1) + (z8-z4) + (z6-z2) + (z7-z3) );


  b.dEdx(i,j,k) = (dydN*dzdX - dydX*dzdN)/ b.J(i,j,k);
  b.dEdy(i,j,k) = (dxdN*dzdX - dxdX*dzdN)/-b.J(i,j,k);
  b.dEdz(i,j,k) = (dxdN*dydX - dxdX*dydN)/ b.J(i,j,k);

  b.dNdx(i,j,k) = (dydE*dzdX - dydX*dzdE)/-b.J(i,j,k);
  b.dNdy(i,j,k) = (dxdE*dzdX - dxdX*dzdE)/ b.J(i,j,k);
  b.dNdz(i,j,k) = (dxdE*dydX - dxdX*dzdE)/-b.J(i,j,k);

  b.dXdx(i,j,k) = (dydE*dzdN - dydN*dzdE)/ b.J(i,j,k);
  b.dXdy(i,j,k) = (dxdE*dzdN - dxdN*dzdE)/-b.J(i,j,k);
  b.dXdz(i,j,k) = (dxdE*dydN - dxdN*dydE)/ b.J(i,j,k);

  });
}};

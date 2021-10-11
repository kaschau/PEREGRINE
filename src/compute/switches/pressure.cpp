#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <math.h>
#include <numeric>
#include <stdexcept>

void pressure(block_ b) {

  MDRange3 range_cc({1,1,1},{b.ni,b.nj,b.nk});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double s = log(b.q(i,j,k,0) / pow(b.Q(i,j,k,0),b.qh(i,j,k,0)));

  double sip = log(b.q(i+1,j,k,0) / pow(b.Q(i+1,j,k,0),b.qh(i+1,j,k,0)));
  double sim = log(b.q(i-1,j,k,0) / pow(b.Q(i-1,j,k,0),b.qh(i-1,j,k,0)));

  double sjp = log(b.q(i,j+1,k,0) / pow(b.Q(i,j+1,k,0),b.qh(i,j+1,k,0)));
  double sjm = log(b.q(i,j-1,k,0) / pow(b.Q(i,j-1,k,0),b.qh(i,j-1,k,0)));

  double skp = log(b.q(i,j,k+1,0) / pow(b.Q(i,j,k+1,0),b.qh(i,j,k+1,0)));
  double skm = log(b.q(i,j,k-1,0) / pow(b.Q(i,j,k-1,0),b.qh(i,j,k-1,0)));

  double ri = ( s - sim ) / ( sip - s );
  b.phi(i,j,k,0) = 1.0 - std::max( 0.0, std::min(1.0, ri) );
  //b.phi(i,j,k,0) = 1.0 - ( pow(ri,2.0) + ri )/( pow(ri,2.0) + 1.0 );

  // b.phi(i,j,k,0) = std::min(std::max( abs( sip - 2*s + sim ) / ( abs(sip) + 2.0*abs(s) + abs(sim) + 1e-16), 0.0), 1.0);
  // b.phi(i,j,k,1) = std::min(std::max( abs( sjp - 2*s + sjm ) / ( abs(sjp) + 2.0*abs(s) + abs(sjm) + 1e-16), 0.0), 1.0);
  // b.phi(i,j,k,2) = std::min(std::max( abs( skp - 2*s + skm ) / ( abs(skp) + 2.0*abs(s) + abs(skm) + 1e-16), 0.0), 1.0);

  });
}

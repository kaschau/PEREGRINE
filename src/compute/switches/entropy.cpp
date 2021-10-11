#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <math.h>
#include <numeric>
#include <stdexcept>

void entropy(block_ b) {

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

  double rj = ( s - sjm ) / ( sjp - s );
  b.phi(i,j,k,1) = 1.0 - std::max( 0.0, std::min(1.0, rj) );

  double rk = ( s - skm ) / ( skp - s );
  b.phi(i,j,k,2) = 1.0 - std::max( 0.0, std::min(1.0, rk) );
  });
}

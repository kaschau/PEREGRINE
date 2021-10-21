#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <math.h>
#include <numeric>

void entropy(block_ b) {

  MDRange3 range_cc({b.ng,b.ng,b.ng},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1});

  Kokkos::parallel_for("Compute switch from entropy",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double s   = log(b.q(i  ,j,k,0) / pow(b.Q(i  ,j,k,0),b.qh(i  ,j,k,0)));

  double sip = log(b.q(i+1,j,k,0) / pow(b.Q(i+1,j,k,0),b.qh(i+1,j,k,0)));
  double sim = log(b.q(i-1,j,k,0) / pow(b.Q(i-1,j,k,0),b.qh(i-1,j,k,0)));

  double sjp = log(b.q(i,j+1,k,0) / pow(b.Q(i,j+1,k,0),b.qh(i,j+1,k,0)));
  double sjm = log(b.q(i,j-1,k,0) / pow(b.Q(i,j-1,k,0),b.qh(i,j-1,k,0)));

  double skp = log(b.q(i,j,k+1,0) / pow(b.Q(i,j,k+1,0),b.qh(i,j,k+1,0)));
  double skm = log(b.q(i,j,k-1,0) / pow(b.Q(i,j,k-1,0),b.qh(i,j,k-1,0)));

  double ri = abs(sip  - 2.0*    s +      sim)
          / ( abs(sip) + 2.0*abs(s) + abs(sim) + 1.0e-16 );
  b.phi(i,j,k,0) = ri;

  double rj = abs(sjp  - 2.0*    s  +     sjm)
          / ( abs(sjp) + 2.0*abs(s) + abs(sjm) + 1.0e-16 );
  b.phi(i,j,k,1) = rj;

  double rk = abs(skp  - 2.0*    s  +     skm)
          / ( abs(skp) + 2.0*abs(s) + abs(skm) + 1.0e-16 );
  b.phi(i,j,k,2) = rk;

  });
}

void pressure(block_ b) {

  MDRange3 range_cc({b.ng,b.ng,b.ng},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double p   = b.q(i  ,j,k,0) ;

  double pip = b.q(i+1,j,k,0) ;
  double pim = b.q(i-1,j,k,0) ;

  double pjp = b.q(i,j+1,k,0) ;
  double pjm = b.q(i,j-1,k,0) ;

  double pkp = b.q(i,j,k+1,0) ;
  double pkm = b.q(i,j,k-1,0) ;

  double ri = abs( pip - 2.0*p + pim )
            / abs( pip + 2.0*p + pim );
  b.phi(i,j,k,0) = ri;

  double rj = abs( pjp - 2.0*p + pjm )
            / abs( pjp + 2.0*p + pjm );
  b.phi(i,j,k,1) = rj;

  double rk = abs( pkp - 2.0*p + pkm )
            / abs( pkp + 2.0*p + pkm );
  b.phi(i,j,k,2) = rk;

  });
}

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <math.h>
#include <numeric>

void smagorinsky(block_ b) {

  MDRange3 range_cc({b.ng,b.ng,b.ng},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1});

  Kokkos::parallel_for("Compute switch from entropy",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  const double Cs = 0.18;
  const double Prt = 0.9;

  double& u = b.q(i,j,k,1);
  double& v = b.q(i,j,k,2);
  double& w = b.q(i,j,k,3);

  double& dudx = b.dqdx(i,j,k,1);
  double& dudy = b.dqdy(i,j,k,1);
  double& dudz = b.dqdz(i,j,k,1);

  double& dvdx = b.dqdx(i,j,k,2);
  double& dvdy = b.dqdy(i,j,k,2);
  double& dvdz = b.dqdz(i,j,k,2);

  double& dwdx = b.dqdx(i,j,k,3);
  double& dwdy = b.dqdy(i,j,k,3);
  double& dwdz = b.dqdz(i,j,k,3);

  double dukdxk = dudx + dvdy + dwdz;

  double S[3][3];
  S[1][1] = 2.0*dudx - 2.0/3.0*dukdxk;
  S[2][2] = 2.0*dvdy - 2.0/3.0*dukdxk;
  S[3][3] = 2.0*dwdz - 2.0/3.0*dukdxk;

  S[1][2] = dudy + dvdx;
  S[2][1] = S[1][2];
  S[1][3] = dudz + dwdx;
  S[3][1] = S[3][1];
  S[2][3] = dvdz + dwdy;
  S[3][2] = S[2][3];

  double invSij=0.0;
  double usg, vsg, wsg;

  for (int l; l < 3; l++) {
    for (int m; m < 3; m++) {
      invSij += S[l][m];
    }
  }
  invSij = sqrt(0.5*fmax(invSij, 0.0));

  double delta = pow(b.J(i,j,k),1.0/3.0);

  double nusgs = pow(Cs*delta,2.0)*invSij;

  double musgs = nusgs * b.Q(i,j,k,0);

  // Add sgs values to properties
  // viscocity
  b.qt(i,j,k,0) += musgs;
  std::cout << musgs << std::endl;
  // thermal conductivity
  double kappasgs = musgs * b.qh(i,j,k,1) / Prt;
  b.qt(i,j,k,1) += kappasgs;

  });
}

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <math.h>
#include <numeric>

void mixedScaleModel(block_ b) {

  MDRange3 range_cc({b.ng,b.ng,b.ng},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1});

  Kokkos::parallel_for("Compute switch from entropy",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  const double Cm = 0.06;
  const double alpha = 0.5;
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

  double A11 = 2.0*dudx - 2.0/3.0*dukdxk;
  double A22 = 2.0*dvdy - 2.0/3.0*dukdxk;
  double A33 = 2.0*dwdz - 2.0/3.0*dukdxk;

  double A12 = dudy + dvdx;
  double A13 = dudz + dwdx;
  double A23 = dvdz + dwdy;

  double invSij, usg, vsg, wsg;

  invSij = A11*A22 + A22*A33 +A11*A33 - pow(A12,2.0) - pow(A13,2.0) - pow(A23,2.0);

  usg = 1.0/3.0 * ( ( 0.25*b.q(i-1,j,k,1) + 0.5*u + 0.25*b.q(i+1,j,k,1) ) +
                    ( 0.25*b.q(i,j-1,k,1) + 0.5*u + 0.25*b.q(i,j+1,k,1) ) +
                    ( 0.25*b.q(i,j,k-1,1) + 0.5*u + 0.25*b.q(i,j,k+1,1) ) );

  vsg = 1.0/3.0 * ( ( 0.25*b.q(i-1,j,k,2) + 0.5*u + 0.25*b.q(i+1,j,k,2) ) +
                    ( 0.25*b.q(i,j-1,k,2) + 0.5*u + 0.25*b.q(i,j+1,k,2) ) +
                    ( 0.25*b.q(i,j,k-1,2) + 0.5*u + 0.25*b.q(i,j,k+1,2) ) );

  wsg = 1.0/3.0 * ( ( 0.25*b.q(i-1,j,k,3) + 0.5*u + 0.25*b.q(i+1,j,k,3) ) +
                    ( 0.25*b.q(i,j-1,k,3) + 0.5*u + 0.25*b.q(i,j+1,k,3) ) +
                    ( 0.25*b.q(i,j,k-1,3) + 0.5*u + 0.25*b.q(i,j,k+1,3) ) );

  double qc2 = 0.5*(pow(u-usg,2.0) + pow(v-vsg,2.0) + pow(w-wsg,2.0));

  double delta = pow(b.J(i,j,k),1.0/3.0);

  double nusgs = Cm*pow(invSij,alpha)*pow(qc2,(1.0-alpha)/2.0)*pow(delta,1.0+alpha);

  double musgs = nusgs * b.Q(i,j,k,0);

  // Add sgs values to properties
  // viscocity
  b.qt(i,j,k,0) += musgs;
  // thermal conductivity
  double kappasgs = musgs / Prt;
  b.qt(i,j,k,1) += kappasgs;

  });
}


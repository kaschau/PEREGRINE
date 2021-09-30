#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"

void rusanov(block_ b, const thtrdat_ th, const double primary) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({1,1,1},{b.ni+1,b.nj,b.nk});
  Kokkos::parallel_for("i face conv fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {
    double ufR, vfR, wfR;
    double UR;
    double rhoR, pR, ER;

    double ufL, vfL, wfL;
    double UL;
    double rhoL, pL, EL;

    ufR = b.q(i  ,j,k,1);
    vfR = b.q(i  ,j,k,2);
    wfR = b.q(i  ,j,k,3);

    ufL = b.q(i-1,j,k,1);
    vfL = b.q(i-1,j,k,2);
    wfL = b.q(i-1,j,k,3);

    UR = b.inx(i,j,k)*ufR +
         b.iny(i,j,k)*vfR +
         b.inz(i,j,k)*wfR ;
    UL = b.inx(i,j,k)*ufL +
         b.iny(i,j,k)*vfL +
         b.inz(i,j,k)*wfL ;

    rhoR = b.Q(i  ,j,k,0);
    rhoL = b.Q(i-1,j,k,0);

    pR = b.q(i  ,j,k,0);
    pL = b.q(i-1,j,k,0);

    ER = b.Q(i  ,j,k,4);
    EL = b.Q(i-1,j,k,4);

    // wave speed estimate / 2
    double lam = 0.25*( ( b.qh(i,j,k,3) + b.qh(i-1,j,k,3) ) + abs(UR + UL) );

    // Continuity rho*Ui
    double FrhoR,FrhoL;
    FrhoR = UR*rhoR;
    FrhoL = UL*rhoL;
    b.iF(i,j,k,0) = 0.5*( FrhoR + FrhoL - lam*(rhoR - rhoL) ) * b.iS(i,j,k);

    double FUR,FUL;
    // x momentum rho*u*Ui+ p*Ax
    FUR = UR*ufR*rhoR + pR*b.isx(i,j,k)/b.iS(i,j,k);
    FUL = UL*ufL*rhoL + pL*b.isx(i,j,k)/b.iS(i,j,k);
    b.iF(i,j,k,1) = 0.5*( FUR + FUL - lam*(rhoR*ufR - rhoL*ufL) ) * b.iS(i,j,k);

    // y momentum rho*v*Ui+ p*Ay
    FUR = UR*vfR*rhoR + pR*b.isy(i,j,k)/b.iS(i,j,k);
    FUL = UL*vfL*rhoL + pL*b.isy(i,j,k)/b.iS(i,j,k);
    b.iF(i,j,k,2) = 0.5*( FUR + FUL - lam*(rhoR*vfR - rhoL*vfL) ) * b.iS(i,j,k);

    // w momentum rho*w*Ui+ p*Az
    FUR = UR*wfR*rhoR + pR*b.isz(i,j,k)/b.iS(i,j,k);
    FUL = UL*wfL*rhoL + pL*b.isz(i,j,k)/b.iS(i,j,k);
    b.iF(i,j,k,3) = 0.5*( FUR + FUL - lam*(rhoR*wfR - rhoL*wfL) ) * b.iS(i,j,k);

    // Total energy (rhoE+ p)*Ui)
    double FER, FEL;
    FER = UR*(ER + pR);
    FEL = UL*(EL + pL);
    b.iF(i,j,k,4) = 0.5*( FER + FEL - lam*(ER - EL) ) * b.iS(i,j,k);

    // Species
    double FYiR, FYiL;
    double YiR, YiL;
    for (int n=0; n<th.ns-1; n++)
    {
      FYiR = rhoR*b.q(i  ,j,k,5+n)*UR;
      FYiL = rhoL*b.q(i-1,j,k,5+n)*UL;
      YiR = b.Q(i  ,j,k,5+n);
      YiL = b.Q(i-1,j,k,5+n);
      b.iF(i,j,k,5+n) = 0.5*( FYiR + FYiL - lam*(YiR - YiL) ) * b.iS(i,j,k);
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({1,1,1},{b.ni,b.nj+1,b.nk});
  Kokkos::parallel_for("j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {


  });
//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({1,1,1},{b.ni,b.nj,b.nk+1});
  Kokkos::parallel_for("k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {


  });
//-------------------------------------------------------------------------------------------|
// Apply fluxes to cc range
//-------------------------------------------------------------------------------------------|
  MDRange4 range({1,1,1,0},{b.ni,b.nj,b.nk,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {


    // Compute switch on face
    double iFphi  = 0.5 * ( b.phi(i,j,k,0) + b.phi(i-1,j,k,0) );
    double iFphi1 = 0.5 * ( b.phi(i,j,k,0) + b.phi(i+1,j,k,0) );
    double jFphi  = 0.5 * ( b.phi(i,j,k,1) + b.phi(i,j-1,k,1) );
    double jFphi1 = 0.5 * ( b.phi(i,j,k,1) + b.phi(i,j+1,k,1) );
    double kFphi  = 0.5 * ( b.phi(i,j,k,2) + b.phi(i,j,k+1,2) );
    double kFphi1 = 0.5 * ( b.phi(i,j,k,2) + b.phi(i,j,k-1,2) );

    double dPrimary = 2.0*primary - 1.0;

    // Add fluxes to RHS
    // format is F_primary*(1-switch) + F_secondary*(switch)
    b.dQ(i,j,k,l) += ( b.iF(i,j,k,l) * (primary -  iFphi * dPrimary) +
                       b.jF(i,j,k,l) * (primary -  jFphi * dPrimary) +
                       b.kF(i,j,k,l) * (primary -  kFphi * dPrimary) ) / b.J(i,j,k) ;

    b.dQ(i,j,k,l) -= ( b.iF(i+1,j,k,l) * (primary -  iFphi1 * dPrimary) +
                       b.jF(i,j+1,k,l) * (primary -  jFphi1 * dPrimary) +
                       b.kF(i,j,k+1,l) * (primary -  kFphi1 * dPrimary) ) / b.J(i,j,k) ;

  });

};

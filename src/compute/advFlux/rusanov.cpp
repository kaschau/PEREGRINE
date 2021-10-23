#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"

void rusanov(block_ b, const thtrdat_ th) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng,b.ng,b.ng},{b.ni+b.ng, b.nj+b.ng-1, b.nk+b.ng-1});
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

    UR = b.isx(i,j,k)*ufR +
         b.isy(i,j,k)*vfR +
         b.isz(i,j,k)*wfR ;
    UL = b.isx(i,j,k)*ufL +
         b.isy(i,j,k)*vfL +
         b.isz(i,j,k)*wfL ;

    rhoR = b.Q(i  ,j,k,0);
    rhoL = b.Q(i-1,j,k,0);

    pR = b.q(i  ,j,k,0);
    pL = b.q(i-1,j,k,0);

    ER = b.Q(i  ,j,k,4);
    EL = b.Q(i-1,j,k,4);

    // wave speed estimate / 2
    double lam = fmax( abs(UL) + b.qh(i,j,k,3)*b.iS(i,j,k), abs(UR) + b.qh(i-1,j,k,3)*b.iS(i,j,k) );

    // Continuity rho*Ui
    double FrhoR,FrhoL;
    FrhoR = UR*rhoR;
    FrhoL = UL*rhoL;
    b.iF(i,j,k,0) = 0.5*( FrhoR + FrhoL - lam*(rhoR - rhoL) );

    double FUR,FUL;
    // x momentum rho*u*Ui+ p*Ax
    FUR = UR*ufR*rhoR + pR*b.isx(i,j,k);
    FUL = UL*ufL*rhoL + pL*b.isx(i,j,k);
    b.iF(i,j,k,1) = 0.5*( FUR + FUL - lam*(rhoR*ufR - rhoL*ufL) );

    // y momentum rho*v*Ui+ p*Ay
    FUR = UR*vfR*rhoR + pR*b.isy(i,j,k);
    FUL = UL*vfL*rhoL + pL*b.isy(i,j,k);
    b.iF(i,j,k,2) = 0.5*( FUR + FUL - lam*(rhoR*vfR - rhoL*vfL) );

    // w momentum rho*w*Ui+ p*Az
    FUR = UR*wfR*rhoR + pR*b.isz(i,j,k);
    FUL = UL*wfL*rhoL + pL*b.isz(i,j,k);
    b.iF(i,j,k,3) = 0.5*( FUR + FUL - lam*(rhoR*wfR - rhoL*wfL) );

    // Total energy (rhoE+ p)*Ui)
    double FER, FEL;
    FER = UR*(ER + pR);
    FEL = UL*(EL + pL);
    b.iF(i,j,k,4) = 0.5*( FER + FEL - lam*(ER - EL) );

    // Species
    double FYiR, FYiL;
    double YiR, YiL;
    for (int n=0; n<th.ns-1; n++)
    {
      FYiR = rhoR*b.q(i  ,j,k,5+n)*UR;
      FYiL = rhoL*b.q(i-1,j,k,5+n)*UL;
      YiR = b.Q(i  ,j,k,5+n);
      YiL = b.Q(i-1,j,k,5+n);
      b.iF(i,j,k,5+n) = 0.5*( FYiR + FYiL - lam*(YiR - YiL) );
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng, b.nk+b.ng-1});
  Kokkos::parallel_for("j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double ufR, vfR, wfR;
    double VR;
    double rhoR, pR, ER;

    double ufL, vfL, wfL;
    double VL;
    double rhoL, pL, EL;

    ufR = b.q(i,j  ,k,1);
    vfR = b.q(i,j  ,k,2);
    wfR = b.q(i,j  ,k,3);

    ufL = b.q(i,j-1,k,1);
    vfL = b.q(i,j-1,k,2);
    wfL = b.q(i,j-1,k,3);

    VR = b.jsx(i,j,k)*ufR +
         b.jsy(i,j,k)*vfR +
         b.jsz(i,j,k)*wfR ;
    VL = b.jsx(i,j,k)*ufL +
         b.jsy(i,j,k)*vfL +
         b.jsz(i,j,k)*wfL ;

    rhoR = b.Q(i,j  ,k,0);
    rhoL = b.Q(i,j-1,k,0);

    pR = b.q(i,j  ,k,0);
    pL = b.q(i,j-1,k,0);

    ER = b.Q(i,j  ,k,4);
    EL = b.Q(i,j-1,k,4);

    // wave speed estimate / 2
    //double lam = 0.25*( ( b.qh(i,j,k,3) + b.qh(i-1,j,k,3) ) + abs(UR + UL) );
    double lam = fmax( abs(VL) + b.qh(i,j,k,3)*b.jS(i,j,k), abs(VR) + b.qh(i,j-1,k,3)*b.jS(i,j,k) );

    // Continuity rho*Ui
    double FrhoR,FrhoL;
    FrhoR = VR*rhoR;
    FrhoL = VL*rhoL;
    b.jF(i,j,k,0) = 0.5*( FrhoR + FrhoL - lam*(rhoR - rhoL) );

    double FUR,FUL;
    // x momentum rho*u*Ui+ p*Ax
    FUR = VR*ufR*rhoR + pR*b.jsx(i,j,k);
    FUL = VL*ufL*rhoL + pL*b.jsx(i,j,k);
    b.jF(i,j,k,1) = 0.5*( FUR + FUL - lam*(rhoR*ufR - rhoL*ufL) );

    // y momentum rho*v*Ui+ p*Ay
    FUR = VR*vfR*rhoR + pR*b.jsy(i,j,k);
    FUL = VL*vfL*rhoL + pL*b.jsy(i,j,k);
    b.jF(i,j,k,2) = 0.5*( FUR + FUL - lam*(rhoR*vfR - rhoL*vfL) );

    // w momentum rho*w*Ui+ p*Az
    FUR = VR*wfR*rhoR + pR*b.jsz(i,j,k);
    FUL = VL*wfL*rhoL + pL*b.jsz(i,j,k);
    b.jF(i,j,k,3) = 0.5*( FUR + FUL - lam*(rhoR*wfR - rhoL*wfL) );

    // Total energy (rhoE+ p)*Ui)
    double FER, FEL;
    FER = VR*(ER + pR);
    FEL = VL*(EL + pL);
    b.jF(i,j,k,4) = 0.5*( FER + FEL - lam*(ER - EL) );

    // Species
    double FYiR, FYiL;
    double YiR, YiL;
    for (int n=0; n<th.ns-1; n++)
    {
      FYiR = rhoR*b.q(i,j  ,k,5+n)*VR;
      FYiL = rhoL*b.q(i,j-1,k,5+n)*VL;
      YiR = b.Q(i,j  ,k,5+n);
      YiL = b.Q(i,j-1,k,5+n);
      b.jF(i,j,k,5+n) = 0.5*( FYiR + FYiL - lam*(YiR - YiL) );
    }


  });
//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng-1, b.nk+b.ng});
  Kokkos::parallel_for("k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double ufR, vfR, wfR;
    double WR;
    double rhoR, pR, ER;

    double ufL, vfL, wfL;
    double WL;
    double rhoL, pL, EL;

    ufR = b.q(i,j,k  ,1);
    vfR = b.q(i,j,k  ,2);
    wfR = b.q(i,j,k  ,3);

    ufL = b.q(i,j,k-1,1);
    vfL = b.q(i,j,k-1,2);
    wfL = b.q(i,j,k-1,3);

    WR = b.ksx(i,j,k)*ufR +
         b.ksy(i,j,k)*vfR +
         b.ksz(i,j,k)*wfR ;
    WL = b.ksx(i,j,k)*ufL +
         b.ksy(i,j,k)*vfL +
         b.ksz(i,j,k)*wfL ;

    rhoR = b.Q(i,j,k  ,0);
    rhoL = b.Q(i,j,k-1,0);

    pR = b.q(i,j,k  ,0);
    pL = b.q(i,j,k-1,0);

    ER = b.Q(i,j,k  ,4);
    EL = b.Q(i,j,k-1,4);

    // wave speed estimate / 2
    //double lam = 0.25*( ( b.qh(i,j,k,3) + b.qh(i-1,j,k,3) ) + abs(UR + UL) );
    double lam = fmax( abs(WL) + b.qh(i,j,k,3)*b.kS(i,j,k), abs(WR) + b.qh(i,j,k-1,3)*b.kS(i,j,k) );

    // Continuity rho*Ui
    double FrhoR,FrhoL;
    FrhoR = WR*rhoR;
    FrhoL = WL*rhoL;
    b.kF(i,j,k,0) = 0.5*( FrhoR + FrhoL - lam*(rhoR - rhoL) );

    double FUR,FUL;
    // x momentum rho*u*Ui+ p*Ax
    FUR = WR*ufR*rhoR + pR*b.ksx(i,j,k);
    FUL = WL*ufL*rhoL + pL*b.ksx(i,j,k);
    b.kF(i,j,k,1) = 0.5*( FUR + FUL - lam*(rhoR*ufR - rhoL*ufL) );

    // y momentum rho*v*Ui+ p*Ay
    FUR = WR*vfR*rhoR + pR*b.ksy(i,j,k);
    FUL = WL*vfL*rhoL + pL*b.ksy(i,j,k);
    b.kF(i,j,k,2) = 0.5*( FUR + FUL - lam*(rhoR*vfR - rhoL*vfL) );

    // w momentum rho*w*Ui+ p*Az
    FUR = WR*wfR*rhoR + pR*b.ksz(i,j,k);
    FUL = WL*wfL*rhoL + pL*b.ksz(i,j,k);
    b.kF(i,j,k,3) = 0.5*( FUR + FUL - lam*(rhoR*wfR - rhoL*wfL) );

    // Total energy (rhoE+ p)*Ui)
    double FER, FEL;
    FER = WR*(ER + pR);
    FEL = WL*(EL + pL);
    b.kF(i,j,k,4) = 0.5*( FER + FEL - lam*(ER - EL) );

    // Species
    double FYiR, FYiL;
    double YiR, YiL;
    for (int n=0; n<th.ns-1; n++)
    {
      FYiR = rhoR*b.q(i,j,k  ,5+n)*WR;
      FYiL = rhoL*b.q(i,j,k-1,5+n)*WL;
      YiR = b.Q(i,j,k  ,5+n);
      YiL = b.Q(i,j,k-1,5+n);
      b.kF(i,j,k,5+n) = 0.5*( FYiR + FYiL - lam*(YiR - YiL) );
    }

  });

};

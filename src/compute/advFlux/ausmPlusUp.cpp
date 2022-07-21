#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "math.h"

void ausmPlusUp(block_ b) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng,b.ng,b.ng},{b.ni+b.ng, b.nj+b.ng-1, b.nk+b.ng-1});
  Kokkos::parallel_for("AUSM+UP i face conv fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {
    double ufR, vfR, wfR;
    double UR;
    double rhoR, pR;

    double ufL, vfL, wfL;
    double UL;
    double rhoL, pL;

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

    pR   = b.q(i  ,j,k,0);
    pL   = b.q(i-1,j,k,0);

    double ML, MR, MbarSQ;
    double a12;

    a12 = 0.5*(b.qh(i,j,k,3) + b.qh(i-1,j,k,3));
    ML = UL/a12;
    MR = UR/a12;

    MbarSQ = (pow(UR,2.0) + pow(UL,2.0)) / (2.0*pow(a12,2.0));

    double Mo;
    const double MinfSQ = 0.1;
    Mo = sqrt( fmin(1.0,fmax(MbarSQ,MinfSQ)) );
    double fa = Mo*(2.0-Mo);

    double M1Plus, M1Minus;
    double M2Plus, M2Minus;
    double M4Plus, M4Minus;
    const double beta = 1.0/8.0;

    M1Plus  = 0.5*(ML+abs(ML));
    M1Minus = 0.5*(MR-abs(MR));
    M2Plus  = 0.25*pow(ML + 1.0, 2.0);
    M2Minus =-0.25*pow(MR - 1.0, 2.0);
    M4Plus  = ( abs(ML) >= 1.0 ) ? M1Plus  : M2Plus *(1.0-16.0*beta*M2Minus) ;
    M4Minus = ( abs(MR) >= 1.0 ) ? M1Minus : M2Minus*(1.0+16.0*beta*M2Plus) ;

    double M12;
    const double Kp = 0.25;
    const double sigma = 1.0;
    double rho12 = 0.5*(rhoR + rhoL);

    M12 = M4Plus + M4Minus - Kp/fa*(fmax(1.0-sigma*MbarSQ, 0.0)) * (pR-pL)/(rho12*pow(a12,2.0));

    double mDot12 = (M12 > 0.0) ? a12*M12*rhoL : a12*M12*rhoR;

    double p5Plus, p5Minus;
    const double alpha = 3.0/16.0*(-4.0 + 5.0*pow(fa,2.0));
    p5Plus  = (abs(ML) >= 1.0) ? 1.0/ML * M1Plus  : M2Plus *(( 2.0-ML)-16.0*alpha*ML*M2Minus);
    p5Minus = (abs(MR) >= 1.0) ? 1.0/MR * M1Minus : M2Minus*((-2.0-MR)+16.0*alpha*MR*M2Plus);

    double p12;
    const double Ku = 0.75;
    p12 = p5Plus*pL + p5Minus*pR - Ku*p5Plus*p5Minus*(rhoR+rhoL)*(fa*a12)*(UR-UL);

    // Upwind the flux
    const int indx = (mDot12 > 0.0) ? -1 : 0;
    // Continuity rho*Ui
    b.iF(i,j,k,0) = mDot12*b.iS(i,j,k);

    // x momentum rho*u*Ui+ p*Ax
    b.iF(i,j,k,1) = mDot12 * b.q(i+indx,j,k,1) * b.iS(i,j,k) + p12*b.isx(i,j,k) ;

    // y momentum rho*v*Ui+ p*Ay
    b.iF(i,j,k,2) = mDot12 * b.q(i+indx,j,k,2) * b.iS(i,j,k) + p12*b.isy(i,j,k) ;

    // w momentum rho*w*Ui+ p*Az
    b.iF(i,j,k,3) = mDot12 * b.q(i+indx,j,k,3) * b.iS(i,j,k) + p12*b.isz(i,j,k) ;

    // Total energy (rhoE+ p)*Ui)
    b.iF(i,j,k,4) = mDot12* (b.Q(i+indx,j,k,4) + b.q(i+indx,j,k,0))/b.Q(i+indx,j,k,0) * b.iS(i,j,k);

    // Species
    for (int n=0; n<b.ne-5; n++)
    {
      b.iF(i,j,k,5+n) = mDot12 * b.q(i+indx,j,k,5+n) * b.iS(i,j,k);
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng, b.nk+b.ng-1});
  Kokkos::parallel_for("AUSM+UP j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double ufR, vfR, wfR;
    double VR;
    double rhoR, pR;

    double ufL, vfL, wfL;
    double VL;
    double rhoL, pL;

    ufR = b.q(i,j  ,k,1);
    vfR = b.q(i,j  ,k,2);
    wfR = b.q(i,j  ,k,3);

    ufL = b.q(i,j-1,k,1);
    vfL = b.q(i,j-1,k,2);
    wfL = b.q(i,j-1,k,3);

    VR = b.jnx(i,j,k)*ufR +
         b.jny(i,j,k)*vfR +
         b.jnz(i,j,k)*wfR ;
    VL = b.jnx(i,j,k)*ufL +
         b.jny(i,j,k)*vfL +
         b.jnz(i,j,k)*wfL ;

    rhoR = b.Q(i,j  ,k,0);
    rhoL = b.Q(i,j-1,k,0);

    pR   = b.q(i,j  ,k,0);
    pL   = b.q(i,j-1,k,0);

    double ML, MR, MbarSQ;
    double a12;

    a12 = 0.5*(b.qh(i,j,k,3) + b.qh(i,j-1,k,3));
    ML = VL/a12;
    MR = VR/a12;

    MbarSQ = (pow(VR,2.0) + pow(VL,2.0)) / (2.0*pow(a12,2.0));

    double Mo;
    const double MinfSQ = 0.1;
    Mo = sqrt( fmin(1.0,fmax(MbarSQ,MinfSQ)) );
    double fa = Mo*(2.0-Mo);

    double M1Plus, M1Minus;
    double M2Plus, M2Minus;
    double M4Plus, M4Minus;
    const double beta = 1.0/8.0;

    M1Plus  = 0.5*(ML+abs(ML));
    M1Minus = 0.5*(MR-abs(MR));
    M2Plus  = 0.25*pow(ML + 1.0, 2.0);
    M2Minus =-0.25*pow(MR - 1.0, 2.0);
    M4Plus  = ( abs(ML) >= 1.0 ) ? M1Plus  : M2Plus *(1.0-16.0*beta*M2Minus) ;
    M4Minus = ( abs(MR) >= 1.0 ) ? M1Minus : M2Minus*(1.0+16.0*beta*M2Plus) ;

    double M12;
    const double Kp = 0.25;
    const double sigma = 1.0;
    double rho12 = 0.5*(rhoR + rhoL);

    M12 = M4Plus + M4Minus - Kp/fa*(fmax(1.0-sigma*MbarSQ, 0.0)) * (pR-pL)/(rho12*pow(a12,2.0));

    double mDot12 = (M12 > 0.0) ? a12*M12*rhoL : a12*M12*rhoR;

    double p5Plus, p5Minus;
    const double alpha = 3.0/16.0*(-4.0 + 5.0*pow(fa,2.0));
    p5Plus  = (abs(ML) >= 1.0) ? 1.0/ML * M1Plus  : M2Plus *(( 2.0-ML)-16.0*alpha*ML*M2Minus);
    p5Minus = (abs(MR) >= 1.0) ? 1.0/MR * M1Minus : M2Minus*((-2.0-MR)+16.0*alpha*MR*M2Plus);

    double p12;
    const double Ku = 0.75;
    p12 = p5Plus*pL + p5Minus*pR - Ku*p5Plus*p5Minus*(rhoR+rhoL)*(fa*a12)*(VR-VL);

    // Upwind the flux
    const int indx = (mDot12 > 0.0) ? -1 : 0;
    // Continuity rho*Ui
    b.jF(i,j,k,0) = mDot12*b.jS(i,j,k);

    // x momentum rho*u*Ui+ p*Ax
    b.jF(i,j,k,1) = mDot12 * b.q(i,j+indx,k,1) * b.jS(i,j,k) + p12*b.jsx(i,j,k) ;

    // y momentum rho*v*Ui+ p*Ay
    b.jF(i,j,k,2) = mDot12 * b.q(i,j+indx,k,2) * b.jS(i,j,k) + p12*b.jsy(i,j,k) ;

    // w momentum rho*w*Ui+ p*Az
    b.jF(i,j,k,3) = mDot12 * b.q(i,j+indx,k,3) * b.jS(i,j,k) + p12*b.jsz(i,j,k) ;

    // Total energy (rhoE+ p)*Ui)
    b.jF(i,j,k,4) = mDot12* (b.Q(i,j+indx,k,4) + b.q(i,j+indx,k,0))/b.Q(i,j+indx,k,0) * b.jS(i,j,k);

    // Species
    for (int n=0; n<b.ne-5; n++)
    {
      b.jF(i,j,k,5+n) = mDot12 * b.q(i,j+indx,k,5+n) * b.jS(i,j,k);
    }


  });
//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng-1, b.nk+b.ng});
  Kokkos::parallel_for("AUSM+UP k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double ufR, vfR, wfR;
    double WR;
    double rhoR, pR;

    double ufL, vfL, wfL;
    double WL;
    double rhoL, pL;

    ufR = b.q(i,j,k  ,1);
    vfR = b.q(i,j,k  ,2);
    wfR = b.q(i,j,k  ,3);

    ufL = b.q(i,j,k-1,1);
    vfL = b.q(i,j,k-1,2);
    wfL = b.q(i,j,k-1,3);

    WR = b.knx(i,j,k)*ufR +
         b.kny(i,j,k)*vfR +
         b.knz(i,j,k)*wfR ;
    WL = b.knx(i,j,k)*ufL +
         b.kny(i,j,k)*vfL +
         b.knz(i,j,k)*wfL ;

    rhoR = b.Q(i,j,k  ,0);
    rhoL = b.Q(i,j,k-1,0);

    pR   = b.q(i,j,k  ,0);
    pL   = b.q(i,j,k-1,0);

    double ML, MR, MbarSQ;
    double a12;

    a12 = 0.5*(b.qh(i,j,k,3) + b.qh(i,j,k-1,3));
    ML = WL/a12;
    MR = WR/a12;

    MbarSQ = (pow(WR,2.0) + pow(WL,2.0)) / (2.0*pow(a12,2.0));

    double Mo;
    const double MinfSQ = 0.1;
    Mo = sqrt( fmin(1.0,fmax(MbarSQ,MinfSQ)) );
    double fa = Mo*(2.0-Mo);

    double M1Plus, M1Minus;
    double M2Plus, M2Minus;
    double M4Plus, M4Minus;
    const double beta = 1.0/8.0;

    M1Plus  = 0.5*(ML+abs(ML));
    M1Minus = 0.5*(MR-abs(MR));
    M2Plus  = 0.25*pow(ML + 1.0, 2.0);
    M2Minus =-0.25*pow(MR - 1.0, 2.0);
    M4Plus  = ( abs(ML) >= 1.0 ) ? M1Plus  : M2Plus *(1.0-16.0*beta*M2Minus) ;
    M4Minus = ( abs(MR) >= 1.0 ) ? M1Minus : M2Minus*(1.0+16.0*beta*M2Plus) ;

    double M12;
    const double Kp = 0.25;
    const double sigma = 1.0;
    double rho12 = 0.5*(rhoR + rhoL);

    M12 = M4Plus + M4Minus - Kp/fa*(fmax(1.0-sigma*MbarSQ, 0.0)) * (pR-pL)/(rho12*pow(a12,2.0));

    double mDot12 = (M12 > 0.0) ? a12*M12*rhoL : a12*M12*rhoR;

    double p5Plus, p5Minus;
    const double alpha = 3.0/16.0*(-4.0 + 5.0*pow(fa,2.0));
    p5Plus  = (abs(ML) >= 1.0) ? 1.0/ML * M1Plus  : M2Plus *(( 2.0-ML)-16.0*alpha*ML*M2Minus);
    p5Minus = (abs(MR) >= 1.0) ? 1.0/MR * M1Minus : M2Minus*((-2.0-MR)+16.0*alpha*MR*M2Plus);

    double p12;
    const double Ku = 0.75;
    p12 = p5Plus*pL + p5Minus*pR - Ku*p5Plus*p5Minus*(rhoR+rhoL)*(fa*a12)*(WR-WL);

    // Upwind the flux
    const int indx = (mDot12 > 0.0) ? -1 : 0;
    // Continuity rho*Ui
    b.kF(i,j,k,0) = mDot12*b.kS(i,j,k);

    // x momentum rho*u*Ui+ p*Ax
    b.kF(i,j,k,1) = mDot12 * b.q(i,j,k+indx,1) * b.kS(i,j,k) + p12*b.ksx(i,j,k) ;

    // y momentum rho*v*Ui+ p*Ay
    b.kF(i,j,k,2) = mDot12 * b.q(i,j,k+indx,2) * b.kS(i,j,k) + p12*b.ksy(i,j,k) ;

    // w momentum rho*w*Ui+ p*Az
    b.kF(i,j,k,3) = mDot12 * b.q(i,j,k+indx,3) * b.kS(i,j,k) + p12*b.ksz(i,j,k) ;

    // Total energy (rhoE+ p)*Ui)
    b.kF(i,j,k,4) = mDot12* (b.Q(i,j,k+indx,4) + b.q(i,j,k+indx,0))/b.Q(i,j,k+indx,0) * b.kS(i,j,k);

    // Species
    for (int n=0; n<b.ne-5; n++)
    {
      b.kF(i,j,k,5+n) = mDot12 * b.q(i,j,k+indx,5+n) * b.kS(i,j,k);
    }

  });

}

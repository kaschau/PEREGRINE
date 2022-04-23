#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"

void hllc(block_ b, const thtrdat_ th) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "rusanov i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - 1, j, k, 1);
        double &vfL = b.q(i - 1, j, k, 2);
        double &wfL = b.q(i - 1, j, k, 3);

        double UR = b.inx(i, j, k) * ufR + b.iny(i, j, k) * vfR + b.inz(i, j, k) * wfR;
        double UL = b.inx(i, j, k) * ufL + b.iny(i, j, k) * vfL + b.inz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i - 1, j, k, 0);

        double &rhouR = b.Q(i, j, k, 1);
        double &rhouL = b.Q(i - 1, j, k, 1);
        double &rhovR = b.Q(i, j, k, 2);
        double &rhovL = b.Q(i - 1, j, k, 2);
        double &rhowR = b.Q(i, j, k, 3);
        double &rhowL = b.Q(i - 1, j, k, 3);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i - 1, j, k, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i - 1, j, k, 4);

        double &cR = b.qh(i, j, k, 3);
        double &cL = b.qh(i - 1, j, k, 3);

        double pstar = 0.5*(pL+pR) - 0.5*(UR-UL)*0.5*(rhoL+rhoR)*0.5*(cL+cR);
        pstar = fmax(0.0,pstar);

        // wave speed estimate
        double SL = UL - cL;
        double SR = UR + cR;
        double Sstar = ( pR - pL + rhoL*UL*(SL-UL) - rhoR*UR*(SR-UR) ) / ( rhoL*(SL-UL) - rhoR*(SR-UR) );

        if ( SL >= 0.0 ){
          b.iF(i, j, k, 0) = UL * rhoL  * b.iS(i, j, k);
          b.iF(i, j, k, 1) = UL * rhouL * b.iS(i, j, k) + pL * b.isx(i, j, k);
          b.iF(i, j, k, 2) = UL * rhovL * b.iS(i, j, k) + pL * b.isy(i, j, k);
          b.iF(i, j, k, 3) = UL * rhowL * b.iS(i, j, k) + pL * b.isz(i, j, k);
          b.iF(i, j, k, 4) = UL * (EL + pL) * b.iS(i, j, k) ;
          for (int n = 0; n < th.ns - 1; n++) {
            double rhoYiL = b.Q(i - 1, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = UL * rhoYiL * b.iS(i, j, k);
          }
        }else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL,FUL,FVL,FWL,FEL, UstarL;
          FrhoL = UL * rhoL  * b.iS(i, j, k);
          FUL   = UL * rhouL * b.iS(i, j, k) + pL * b.isx(i, j, k);
          FVL   = UL * rhovL * b.iS(i, j, k) + pL * b.isy(i, j, k);
          FWL   = UL * rhowL * b.iS(i, j, k) + pL * b.isz(i, j, k);
          FEL   = UL * (EL + pL) * b.iS(i, j, k);
          UstarL = rhoL * (SL - UL)/(SL - Sstar);

          b.iF(i, j, k, 0) = FrhoL + SL*(UstarL - rhoL) * b.iS(i, j, k);
          b.iF(i, j, k, 1) = FUL   + SL*(UstarL*ufL - rhouL) * b.iS(i, j, k);
          b.iF(i, j, k, 2) = FVL   + SL*(UstarL*vfL - rhovL) * b.iS(i, j, k);
          b.iF(i, j, k, 3) = FWL   + SL*(UstarL*wfL - rhowL) * b.iS(i, j, k);
          b.iF(i, j, k, 4) = FEL   + SL*(UstarL*(EL/rhoL + (Sstar - UL)*(Sstar + pL/(rhoL*(SL-UL)))) - EL) * b.iS(i, j, k);
          for (int n = 0; n < th.ns - 1; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i - 1, j, k, 5 + n) * UL * b.iS(i, j, k);
            YiL = b.q(i - 1, j, k, 5 + n);
            rhoYiL = b.Q(i - 1, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = FYiL + SL*(UstarL*YiL - rhoYiL) * b.iS(i, j, k);
          }
        }else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR,FUR,FVR,FWR,FER, UstarR;
          FrhoR = UR * rhoR  * b.iS(i, j, k);
          FUR   = UR * rhouR * b.iS(i, j, k) + pR * b.isx(i, j, k);
          FVR   = UR * rhovR * b.iS(i, j, k) + pR * b.isy(i, j, k);
          FWR   = UR * rhowR * b.iS(i, j, k) + pR * b.isz(i, j, k);
          FER   = UR * (ER + pR) * b.iS(i, j, k);
          UstarR = rhoR * (SR - UR)/(SR - Sstar);

          b.iF(i, j, k, 0) = FrhoR + SR*(UstarR - rhoR) * b.iS(i, j, k);
          b.iF(i, j, k, 1) = FUR   + SR*(UstarR*ufR - rhouR) * b.iS(i, j, k);
          b.iF(i, j, k, 2) = FVR   + SR*(UstarR*vfR - rhovR) * b.iS(i, j, k);
          b.iF(i, j, k, 3) = FWR   + SR*(UstarR*wfR - rhowR) * b.iS(i, j, k);
          b.iF(i, j, k, 4) = FER   + SR*(UstarR*(ER/rhoR + (Sstar - UR)*(Sstar + pR/(rhoR*(SR-UR)))) - ER) * b.iS(i, j, k);
          for (int n = 0; n < th.ns - 1; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * b.iS(i, j, k);
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = FYiR + SR*(UstarR*YiR - rhoYiR) * b.iS(i, j, k);
          }
        }else if (SR <= 0.0) {
          b.iF(i, j, k, 0) = UR * rhoR  * b.iS(i, j, k);
          b.iF(i, j, k, 1) = UR * rhouR * b.iS(i, j, k) + pR * b.isx(i, j, k);
          b.iF(i, j, k, 2) = UR * rhovR * b.iS(i, j, k) + pR * b.isy(i, j, k);
          b.iF(i, j, k, 3) = UR * rhowR * b.iS(i, j, k) + pR * b.isz(i, j, k);
          b.iF(i, j, k, 4) = UR * (ER + pR);
          for (int n = 0; n < th.ns - 1; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = UR * rhoYiR * b.iS(i, j, k);
          }
        }
      });

  // //-------------------------------------------------------------------------------------------|
  // // j flux face range
  // //-------------------------------------------------------------------------------------------|
  // MDRange3 range_j({b.ng, b.ng, b.ng},
  //                  {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  // Kokkos::parallel_for(
  //     "rusanov j face conv fluxes", range_j,
  //     KOKKOS_LAMBDA(const int i, const int j, const int k) {

  //     });
  // //-------------------------------------------------------------------------------------------|
  // // k flux face range
  // //-------------------------------------------------------------------------------------------|
  // MDRange3 range_k({b.ng, b.ng, b.ng},
  //                  {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  // Kokkos::parallel_for(
  //     "rusanov k face conv fluxes", range_k,
  //     KOKKOS_LAMBDA(const int i, const int j, const int k) {

  //     });
}

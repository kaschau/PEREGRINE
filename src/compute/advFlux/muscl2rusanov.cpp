#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"
#include "math.h"

// Compute the flux at a face using 2nd order MUSCL reconstruction with rusanov flux at face
//
// Here is the strategy:
//
//      Reconstruct   rho, u,v,w e(internal)
//      Use the reconstruction of e for p and c (speed of sound)
//      Use the reconstruction of rho for Y
//
//      Use reconstruction of u,v,w to compute kinetic energy at face,
//      then use KE and e to compute total energy at face.
//
//      For the slope limiter, we use the generalized minmod limiter
//      where we can adjust theta E[1,2] where theta=1 is most dissipative
//      and theta=2 is least dissipative (according to wikipedia)


void muscl2rusanov(block_ b, const thtrdat_ th) {

  double theta = 2.0;
  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

        double rR,rL, phiR,phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi   = b.Q(i  ,j ,k ,0);
        double &rhoim1 = b.Q(i-1,j ,k ,0);
        double &rhoim2 = b.Q(i-2,j ,k ,0);
        double &rhoip1 = b.Q(i+1,j ,k ,0);
        rR = (rhoi - rhoim1)/(rhoip1-rhoi);
        rL = (rhoim1 - rhoim2)/(rhoi-rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double rhoR = rhoi   - 0.5*phiRhoR*(rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5*phiRhoL*(rhoi - rhoim1);

        // Reconstruct u
        double &ui   = b.q(i  ,j ,k ,1);
        double &uim1 = b.q(i-1,j ,k ,1);
        double &uim2 = b.q(i-2,j ,k ,1);
        double &uip1 = b.q(i+1,j ,k ,1);
        rR = (ui - uim1)/(uip1-ui);
        rL = (uim1 - uim2)/(ui-uim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double ufR = ui   - 0.5*phiR*(uip1 - ui);
        double ufL = uim1 + 0.5*phiL*(ui   - uim1);

        // Reconstruct v
        double &vi   = b.q(i  ,j ,k ,2);
        double &vim1 = b.q(i-1,j ,k ,2);
        double &vim2 = b.q(i-2,j ,k ,2);
        double &vip1 = b.q(i+1,j ,k ,2);
        rR = (vi - vim1)/(vip1-vi);
        rL = (vim1 - vim2)/(vi-vim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double vfR = vi   - 0.5*phiR*(vip1 - vi);
        double vfL = vim1 + 0.5*phiL*(vi   - vim1);

        // Reconstruct w
        double &wi   = b.q(i  ,j ,k ,3);
        double &wim1 = b.q(i-1,j ,k ,3);
        double &wim2 = b.q(i-2,j ,k ,3);
        double &wip1 = b.q(i+1,j ,k ,3);
        rR = (wi - wim1)/(wip1-wi);
        rL = (wim1 - wim2)/(wi-wim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double wfR = wi   - 0.5*phiR*(wip1 - wi);
        double wfL = wim1 + 0.5*phiL*(wi   - wim1);

        // Face normal velocity
        double UR = b.isx(i, j, k) * ufR + b.isy(i, j, k) * vfR + b.isz(i, j, k) * wfR;
        double UL = b.isx(i, j, k) * ufL + b.isy(i, j, k) * vfL + b.isz(i, j, k) * wfL;

        // Reconstruct e
        double ei   = b.qh(i  ,j ,k ,4)/rhoi;
        double eim1 = b.qh(i-1,j ,k ,4)/rhoim1;
        double eim2 = b.qh(i-2,j ,k ,4)/rhoim2;
        double eip1 = b.qh(i+1,j ,k ,4)/rhoip1;
        rR = (ei - eim1)/(eip1-ei);
        rL = (eim1 - eim2)/(ei-eim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double eR = ei   - 0.5*phiR*(eip1 - ei);
        double eL = eim1 + 0.5*phiL*(ei   - eim1);

        // Reuse reconstruction for p, c
        double &pi   = b.q(i  ,j ,k ,0);
        double &pim1 = b.q(i-1,j ,k ,0);
        //double &pim2 = b.q(i-2,j ,k ,0);
        double &pip1 = b.q(i+1,j ,k ,0);
        double pR = pi   - 0.5*phiR*(pip1 - pi);
        double pL = pim1 + 0.5*phiL*(pi   - pim1);

        double &ci   = b.qh(i  ,j ,k ,3);
        double &cim1 = b.qh(i-1,j ,k ,3);
        //double &cim2 = b.qh(i-2,j ,k ,3);
        double &cip1 = b.qh(i+1,j ,k ,3);
        double cR = ci   - 0.5*phiR*(cip1 - ci);
        double cL = cim1 + 0.5*phiL*(ci   - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5* (pow(ufR,2.0) + pow(vfR,2.0) + pow(wfR,2.0));
        double kL = 0.5* (pow(ufL,2.0) + pow(vfL,2.0) + pow(wfL,2.0));
        double ER = rhoR*(eR + kR);
        double EL = rhoL*(eL + kL);


        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL,
                          abs(UR) + cR) * b.iS(i, j, k);

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        b.iF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * b.isx(i, j, k);
        FUL = UL * ufL * rhoL + pL * b.isx(i, j, k);
        b.iF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * b.isy(i, j, k);
        FUL = UL * vfL * rhoL + pL * b.isy(i, j, k);
        b.iF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * b.isz(i, j, k);
        FUL = UL * wfL * rhoL + pL * b.isz(i, j, k);
        b.iF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        b.iF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < th.ns - 1; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi   = b.Q(i  ,j ,k ,5+n);
          double &rhoYim1 = b.Q(i-1,j ,k ,5+n);
          double &rhoYip1 = b.Q(i+1,j ,k ,5+n);
          rhoYiR = rhoYi   - 0.5*phiRhoR*(rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5*phiRhoL*(rhoYi   - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          b.iF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

        double rR,rL, phiR,phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi   = b.Q(i ,j   ,k ,0);
        double &rhoim1 = b.Q(i ,j-1 ,k ,0);
        double &rhoim2 = b.Q(i ,j-2 ,k ,0);
        double &rhoip1 = b.Q(i ,j+1 ,k ,0);
        rR = (rhoi - rhoim1)/(rhoip1-rhoi);
        rL = (rhoim1 - rhoim2)/(rhoi-rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double rhoR = rhoi   - 0.5*phiRhoR*(rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5*phiRhoL*(rhoi - rhoim1);

        // Reconstruct u
        double &ui   = b.q(i ,j   ,k ,1);
        double &uim1 = b.q(i ,j-1 ,k ,1);
        double &uim2 = b.q(i ,j-2 ,k ,1);
        double &uip1 = b.q(i ,j+1 ,k ,1);
        rR = (ui - uim1)/(uip1-ui);
        rL = (uim1 - uim2)/(ui-uim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double ufR = ui   - 0.5*phiR*(uip1 - ui);
        double ufL = uim1 + 0.5*phiL*(ui   - uim1);

        // Reconstruct v
        double &vi   = b.q(i, j  ,k ,2);
        double &vim1 = b.q(i, j-1,k ,2);
        double &vim2 = b.q(i, j-2,k ,2);
        double &vip1 = b.q(i, j+1,k ,2);
        rR = (vi - vim1)/(vip1-vi);
        rL = (vim1 - vim2)/(vi-vim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double vfR = vi   - 0.5*phiR*(vip1 - vi);
        double vfL = vim1 + 0.5*phiL*(vi   - vim1);

        // Reconstruct w
        double &wi   = b.q(i ,j   ,k ,3);
        double &wim1 = b.q(i ,j-1 ,k ,3);
        double &wim2 = b.q(i ,j-2 ,k ,3);
        double &wip1 = b.q(i ,j+1 ,k ,3);
        rR = (wi - wim1)/(wip1-wi);
        rL = (wim1 - wim2)/(wi-wim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double wfR = wi   - 0.5*phiR*(wip1 - wi);
        double wfL = wim1 + 0.5*phiL*(wi   - wim1);

        // Face normal velocity
        double UR = b.jsx(i, j, k) * ufR + b.jsy(i, j, k) * vfR + b.jsz(i, j, k) * wfR;
        double UL = b.jsx(i, j, k) * ufL + b.jsy(i, j, k) * vfL + b.jsz(i, j, k) * wfL;

        // Reconstruct e
        double ei   = b.qh(i ,j   ,k ,4)/rhoi;
        double eim1 = b.qh(i ,j-1 ,k ,4)/rhoim1;
        double eim2 = b.qh(i ,j-2 ,k ,4)/rhoim2;
        double eip1 = b.qh(i ,j+1 ,k ,4)/rhoip1;
        rR = (ei - eim1)/(eip1-ei);
        rL = (eim1 - eim2)/(ei-eim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double eR = ei   - 0.5*phiR*(eip1 - ei);
        double eL = eim1 + 0.5*phiL*(ei   - eim1);

        // Reuse reconstruction for p, c
        double &pi   = b.q(i ,j   ,k ,0);
        double &pim1 = b.q(i ,j-1 ,k ,0);
        //double &pim2 = b.q(i ,j-2 ,k ,0);
        double &pip1 = b.q(i ,j+1 ,k ,0);
        double pR = pi   - 0.5*phiR*(pip1 - pi);
        double pL = pim1 + 0.5*phiL*(pi   - pim1);

        double &ci   = b.qh(i ,j   ,k ,3);
        double &cim1 = b.qh(i ,j-1 ,k ,3);
        //double &cim2 = b.qh(i ,j-2 ,k ,3);
        double &cip1 = b.qh(i ,j+1 ,k ,3);
        double cR = ci   - 0.5*phiR*(cip1 - ci);
        double cL = cim1 + 0.5*phiL*(ci   - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5* (pow(ufR,2.0) + pow(vfR,2.0) + pow(wfR,2.0));
        double kL = 0.5* (pow(ufL,2.0) + pow(vfL,2.0) + pow(wfL,2.0));
        double ER = rhoR*(eR + kR);
        double EL = rhoL*(eL + kL);

        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL,
                          abs(UR) + cR) * b.jS(i, j, k);

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        b.jF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * b.jsx(i, j, k);
        FUL = UL * ufL * rhoL + pL * b.jsx(i, j, k);
        b.jF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * b.jsy(i, j, k);
        FUL = UL * vfL * rhoL + pL * b.jsy(i, j, k);
        b.jF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * b.jsz(i, j, k);
        FUL = UL * wfL * rhoL + pL * b.jsz(i, j, k);
        b.jF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        b.jF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < th.ns - 1; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi   = b.Q(i ,j   ,k ,5+n);
          double &rhoYim1 = b.Q(i ,j-1 ,k ,5+n);
          double &rhoYip1 = b.Q(i ,j+1 ,k ,5+n);
          rhoYiR = rhoYi   - 0.5*phiRhoR*(rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5*phiRhoL*(rhoYi   - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          b.jF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });
  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

        double rR,rL, phiR,phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi   = b.Q(i ,j ,k   ,0);
        double &rhoim1 = b.Q(i ,j ,k-1 ,0);
        double &rhoim2 = b.Q(i ,j ,k-2 ,0);
        double &rhoip1 = b.Q(i ,j ,k+1 ,0);
        rR = (rhoi - rhoim1)/(rhoip1-rhoi);
        rL = (rhoim1 - rhoim2)/(rhoi-rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double rhoR = rhoi   - 0.5*phiRhoR*(rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5*phiRhoL*(rhoi - rhoim1);

        // Reconstruct u
        double &ui   = b.q(i ,j ,k   ,1);
        double &uim1 = b.q(i ,j ,k-1 ,1);
        double &uim2 = b.q(i ,j ,k-2 ,1);
        double &uip1 = b.q(i ,j ,k+1 ,1);
        rR = (ui - uim1)/(uip1-ui);
        rL = (uim1 - uim2)/(ui-uim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double ufR = ui   - 0.5*phiR*(uip1 - ui);
        double ufL = uim1 + 0.5*phiL*(ui   - uim1);

        // Reconstruct v
        double &vi   = b.q(i ,j ,k   ,2);
        double &vim1 = b.q(i ,j ,k-1 ,2);
        double &vim2 = b.q(i ,j ,k-2 ,2);
        double &vip1 = b.q(i ,j ,k+1 ,2);
        rR = (vi - vim1)/(vip1-vi);
        rL = (vim1 - vim2)/(vi-vim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double vfR = vi   - 0.5*phiR*(vip1 - vi);
        double vfL = vim1 + 0.5*phiL*(vi   - vim1);

        // Reconstruct w
        double &wi   = b.q(i ,j ,k   ,3);
        double &wim1 = b.q(i ,j ,k-1 ,3);
        double &wim2 = b.q(i ,j ,k-2 ,3);
        double &wip1 = b.q(i ,j ,k+1 ,3);
        rR = (wi - wim1)/(wip1-wi);
        rL = (wim1 - wim2)/(wi-wim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double wfR = wi   - 0.5*phiR*(wip1 - wi);
        double wfL = wim1 + 0.5*phiL*(wi   - wim1);

        // Face normal velocity
        double UR = b.ksx(i, j, k) * ufR + b.ksy(i, j, k) * vfR + b.ksz(i, j, k) * wfR;
        double UL = b.ksx(i, j, k) * ufL + b.ksy(i, j, k) * vfL + b.ksz(i, j, k) * wfL;

        // Reconstruct e
        double ei   = b.qh(i ,j ,k   ,4)/rhoi;
        double eim1 = b.qh(i ,j ,k-1 ,4)/rhoim1;
        double eim2 = b.qh(i ,j ,k-2 ,4)/rhoim2;
        double eip1 = b.qh(i ,j ,k+1 ,4)/rhoip1;
        rR = (ei - eim1)/(eip1-ei);
        rL = (eim1 - eim2)/(ei-eim1);
        phiR = fmax(0.0, fmin(fmin(theta*rR, (1.0+rR)/2.0),theta));
        phiL = fmax(0.0, fmin(fmin(theta*rL, (1.0+rL)/2.0),theta));

        double eR = ei   - 0.5*phiR*(eip1 - ei);
        double eL = eim1 + 0.5*phiL*(ei   - eim1);

        // Reuse reconstruction for p, c
        double &pi   = b.q(i ,j ,k   ,0);
        double &pim1 = b.q(i ,j ,k-1 ,0);
        //double &pim2 = b.q(i ,j ,k-2 ,0);
        double &pip1 = b.q(i ,j ,k+1 ,0);
        double pR = pi   - 0.5*phiR*(pip1 - pi);
        double pL = pim1 + 0.5*phiL*(pi   - pim1);

        double &ci   = b.qh(i ,j ,k   ,3);
        double &cim1 = b.qh(i ,j ,k-1 ,3);
        //double &cim2 = b.qh(i ,j ,k-2 ,3);
        double &cip1 = b.qh(i ,j ,k+1 ,3);
        double cR = ci   - 0.5*phiR*(cip1 - ci);
        double cL = cim1 + 0.5*phiL*(ci   - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5* (pow(ufR,2.0) + pow(vfR,2.0) + pow(wfR,2.0));
        double kL = 0.5* (pow(ufL,2.0) + pow(vfL,2.0) + pow(wfL,2.0));
        double ER = rhoR*(eR + kR);
        double EL = rhoL*(eL + kL);

        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL,
                          abs(UR) + cR) * b.kS(i, j, k);

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        b.kF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * b.ksx(i, j, k);
        FUL = UL * ufL * rhoL + pL * b.ksx(i, j, k);
        b.kF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * b.ksy(i, j, k);
        FUL = UL * vfL * rhoL + pL * b.ksy(i, j, k);
        b.kF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * b.ksz(i, j, k);
        FUL = UL * wfL * rhoL + pL * b.ksz(i, j, k);
        b.kF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        b.kF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < th.ns - 1; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi   = b.Q(i ,j ,k   ,5+n);
          double &rhoYim1 = b.Q(i ,j ,k-1 ,5+n);
          double &rhoYip1 = b.Q(i ,j ,k+1 ,5+n);
          rhoYiR = rhoYi   - 0.5*phiRhoR*(rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5*phiRhoL*(rhoYi   - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          b.kF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });
}

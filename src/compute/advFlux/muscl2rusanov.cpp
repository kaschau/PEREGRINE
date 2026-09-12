#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

// Compute the flux at a face using 2nd order MUSCL reconstruction with rusanov
// flux at face
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

PG_ABI void pgMuscl2rusanov(const pgView *Q_, const pgView *iF_,
                            const pgView *iS_, const pgView *jF_,
                            const pgView *jS_, const pgView *kF_,
                            const pgView *kS_, const pgView *q_,
                            const pgView *qh_, const pgDims *d) {
  auto Q = as4(*Q_);
  auto iF = as4(*iF_);
  auto iS = as4(*iS_);
  auto jF = as4(*jF_);
  auto jS = as4(*jS_);
  auto kF = as4(*kF_);
  auto kS = as4(*kS_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;

  double theta = 2.0;
  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({ng, ng, ng}, {ni + ng, nj + ng - 1, nk + ng - 1});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        double rR, rL, phiR, phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi = Q(i, j, k, 0);
        double &rhoim1 = Q(i - 1, j, k, 0);
        double &rhoim2 = Q(i - 2, j, k, 0);
        double &rhoip1 = Q(i + 1, j, k, 0);
        rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
        rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

        // Reconstruct u
        double &ui = q(i, j, k, 1);
        double &uim1 = q(i - 1, j, k, 1);
        double &uim2 = q(i - 2, j, k, 1);
        double &uip1 = q(i + 1, j, k, 1);
        rR = (ui - uim1) / (uip1 - ui);
        rL = (uim1 - uim2) / (ui - uim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double ufR = ui - 0.5 * phiR * (uip1 - ui);
        double ufL = uim1 + 0.5 * phiL * (ui - uim1);

        // Reconstruct v
        double &vi = q(i, j, k, 2);
        double &vim1 = q(i - 1, j, k, 2);
        double &vim2 = q(i - 2, j, k, 2);
        double &vip1 = q(i + 1, j, k, 2);
        rR = (vi - vim1) / (vip1 - vi);
        rL = (vim1 - vim2) / (vi - vim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double vfR = vi - 0.5 * phiR * (vip1 - vi);
        double vfL = vim1 + 0.5 * phiL * (vi - vim1);

        // Reconstruct w
        double &wi = q(i, j, k, 3);
        double &wim1 = q(i - 1, j, k, 3);
        double &wim2 = q(i - 2, j, k, 3);
        double &wip1 = q(i + 1, j, k, 3);
        rR = (wi - wim1) / (wip1 - wi);
        rL = (wim1 - wim2) / (wi - wim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double wfR = wi - 0.5 * phiR * (wip1 - wi);
        double wfL = wim1 + 0.5 * phiL * (wi - wim1);

        // Face normal velocity
        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        // Reconstruct e
        double ei = qh(i, j, k, 4) / rhoi;
        double eim1 = qh(i - 1, j, k, 4) / rhoim1;
        double eim2 = qh(i - 2, j, k, 4) / rhoim2;
        double eip1 = qh(i + 1, j, k, 4) / rhoip1;
        rR = (ei - eim1) / (eip1 - ei);
        rL = (eim1 - eim2) / (ei - eim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double eR = ei - 0.5 * phiR * (eip1 - ei);
        double eL = eim1 + 0.5 * phiL * (ei - eim1);

        // Reuse reconstruction for p, c
        double &pi = q(i, j, k, 0);
        double &pim1 = q(i - 1, j, k, 0);
        // double &pim2 = q(i-2,j ,k ,0);
        double &pip1 = q(i + 1, j, k, 0);
        double pR = pi - 0.5 * phiR * (pip1 - pi);
        double pL = pim1 + 0.5 * phiL * (pi - pim1);

        double &ci = qh(i, j, k, 3);
        double &cim1 = qh(i - 1, j, k, 3);
        // double &cim2 = qh(i-2,j ,k ,3);
        double &cip1 = qh(i + 1, j, k, 3);
        double cR = ci - 0.5 * phiR * (cip1 - ci);
        double cL = cim1 + 0.5 * phiL * (ci - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5 * (pow(ufR, 2.0) + pow(vfR, 2.0) + pow(wfR, 2.0));
        double kL = 0.5 * (pow(ufL, 2.0) + pow(vfL, 2.0) + pow(wfL, 2.0));
        double ER = rhoR * (eR + kR);
        double EL = rhoL * (eL + kL);

        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL, abs(UR) + cR) * S;
        UR *= S;
        UL *= S;

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        iF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * iS(i, j, k, 0);
        FUL = UL * ufL * rhoL + pL * iS(i, j, k, 0);
        iF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * iS(i, j, k, 1);
        FUL = UL * vfL * rhoL + pL * iS(i, j, k, 1);
        iF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * iS(i, j, k, 2);
        FUL = UL * wfL * rhoL + pL * iS(i, j, k, 2);
        iF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        iF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < ne - 5; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi = Q(i, j, k, 5 + n);
          double &rhoYim1 = Q(i - 1, j, k, 5 + n);
          double &rhoYip1 = Q(i + 1, j, k, 5 + n);
          rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          iF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({ng, ng, ng}, {ni + ng - 1, nj + ng, nk + ng - 1});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(jS(i, j, k, 0), jS(i, j, k, 1), jS(i, j, k, 2), S, nx, ny,
                   nz);

        double rR, rL, phiR, phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi = Q(i, j, k, 0);
        double &rhoim1 = Q(i, j - 1, k, 0);
        double &rhoim2 = Q(i, j - 2, k, 0);
        double &rhoip1 = Q(i, j + 1, k, 0);
        rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
        rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

        // Reconstruct u
        double &ui = q(i, j, k, 1);
        double &uim1 = q(i, j - 1, k, 1);
        double &uim2 = q(i, j - 2, k, 1);
        double &uip1 = q(i, j + 1, k, 1);
        rR = (ui - uim1) / (uip1 - ui);
        rL = (uim1 - uim2) / (ui - uim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double ufR = ui - 0.5 * phiR * (uip1 - ui);
        double ufL = uim1 + 0.5 * phiL * (ui - uim1);

        // Reconstruct v
        double &vi = q(i, j, k, 2);
        double &vim1 = q(i, j - 1, k, 2);
        double &vim2 = q(i, j - 2, k, 2);
        double &vip1 = q(i, j + 1, k, 2);
        rR = (vi - vim1) / (vip1 - vi);
        rL = (vim1 - vim2) / (vi - vim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double vfR = vi - 0.5 * phiR * (vip1 - vi);
        double vfL = vim1 + 0.5 * phiL * (vi - vim1);

        // Reconstruct w
        double &wi = q(i, j, k, 3);
        double &wim1 = q(i, j - 1, k, 3);
        double &wim2 = q(i, j - 2, k, 3);
        double &wip1 = q(i, j + 1, k, 3);
        rR = (wi - wim1) / (wip1 - wi);
        rL = (wim1 - wim2) / (wi - wim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double wfR = wi - 0.5 * phiR * (wip1 - wi);
        double wfL = wim1 + 0.5 * phiL * (wi - wim1);

        // Face normal velocity
        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        // Reconstruct e
        double ei = qh(i, j, k, 4) / rhoi;
        double eim1 = qh(i, j - 1, k, 4) / rhoim1;
        double eim2 = qh(i, j - 2, k, 4) / rhoim2;
        double eip1 = qh(i, j + 1, k, 4) / rhoip1;
        rR = (ei - eim1) / (eip1 - ei);
        rL = (eim1 - eim2) / (ei - eim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double eR = ei - 0.5 * phiR * (eip1 - ei);
        double eL = eim1 + 0.5 * phiL * (ei - eim1);

        // Reuse reconstruction for p, c
        double &pi = q(i, j, k, 0);
        double &pim1 = q(i, j - 1, k, 0);
        // double &pim2 = q(i ,j-2 ,k ,0);
        double &pip1 = q(i, j + 1, k, 0);
        double pR = pi - 0.5 * phiR * (pip1 - pi);
        double pL = pim1 + 0.5 * phiL * (pi - pim1);

        double &ci = qh(i, j, k, 3);
        double &cim1 = qh(i, j - 1, k, 3);
        // double &cim2 = qh(i ,j-2 ,k ,3);
        double &cip1 = qh(i, j + 1, k, 3);
        double cR = ci - 0.5 * phiR * (cip1 - ci);
        double cL = cim1 + 0.5 * phiL * (ci - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5 * (pow(ufR, 2.0) + pow(vfR, 2.0) + pow(wfR, 2.0));
        double kL = 0.5 * (pow(ufL, 2.0) + pow(vfL, 2.0) + pow(wfL, 2.0));
        double ER = rhoR * (eR + kR);
        double EL = rhoL * (eL + kL);

        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL, abs(UR) + cR) * S;
        UR *= S;
        UL *= S;

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        jF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * jS(i, j, k, 0);
        FUL = UL * ufL * rhoL + pL * jS(i, j, k, 0);
        jF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * jS(i, j, k, 1);
        FUL = UL * vfL * rhoL + pL * jS(i, j, k, 1);
        jF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * jS(i, j, k, 2);
        FUL = UL * wfL * rhoL + pL * jS(i, j, k, 2);
        jF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        jF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < ne - 5; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi = Q(i, j, k, 5 + n);
          double &rhoYim1 = Q(i, j - 1, k, 5 + n);
          double &rhoYip1 = Q(i, j + 1, k, 5 + n);
          rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          jF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });
  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng});
  Kokkos::parallel_for(
      "MUSCL 2 rusanov k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(kS(i, j, k, 0), kS(i, j, k, 1), kS(i, j, k, 2), S, nx, ny,
                   nz);

        double rR, rL, phiR, phiL;

        // Reconstruct density
        // We store density reconstrution values for species
        double phiRhoR, phiRhoL;
        double &rhoi = Q(i, j, k, 0);
        double &rhoim1 = Q(i, j, k - 1, 0);
        double &rhoim2 = Q(i, j, k - 2, 0);
        double &rhoip1 = Q(i, j, k + 1, 0);
        rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
        rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
        phiRhoR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiRhoL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
        double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

        // Reconstruct u
        double &ui = q(i, j, k, 1);
        double &uim1 = q(i, j, k - 1, 1);
        double &uim2 = q(i, j, k - 2, 1);
        double &uip1 = q(i, j, k + 1, 1);
        rR = (ui - uim1) / (uip1 - ui);
        rL = (uim1 - uim2) / (ui - uim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double ufR = ui - 0.5 * phiR * (uip1 - ui);
        double ufL = uim1 + 0.5 * phiL * (ui - uim1);

        // Reconstruct v
        double &vi = q(i, j, k, 2);
        double &vim1 = q(i, j, k - 1, 2);
        double &vim2 = q(i, j, k - 2, 2);
        double &vip1 = q(i, j, k + 1, 2);
        rR = (vi - vim1) / (vip1 - vi);
        rL = (vim1 - vim2) / (vi - vim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double vfR = vi - 0.5 * phiR * (vip1 - vi);
        double vfL = vim1 + 0.5 * phiL * (vi - vim1);

        // Reconstruct w
        double &wi = q(i, j, k, 3);
        double &wim1 = q(i, j, k - 1, 3);
        double &wim2 = q(i, j, k - 2, 3);
        double &wip1 = q(i, j, k + 1, 3);
        rR = (wi - wim1) / (wip1 - wi);
        rL = (wim1 - wim2) / (wi - wim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double wfR = wi - 0.5 * phiR * (wip1 - wi);
        double wfL = wim1 + 0.5 * phiL * (wi - wim1);

        // Face normal velocity
        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        // Reconstruct e
        double ei = qh(i, j, k, 4) / rhoi;
        double eim1 = qh(i, j, k - 1, 4) / rhoim1;
        double eim2 = qh(i, j, k - 2, 4) / rhoim2;
        double eip1 = qh(i, j, k + 1, 4) / rhoip1;
        rR = (ei - eim1) / (eip1 - ei);
        rL = (eim1 - eim2) / (ei - eim1);
        phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
        phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

        double eR = ei - 0.5 * phiR * (eip1 - ei);
        double eL = eim1 + 0.5 * phiL * (ei - eim1);

        // Reuse reconstruction for p, c
        double &pi = q(i, j, k, 0);
        double &pim1 = q(i, j, k - 1, 0);
        // double &pim2 = q(i ,j ,k-2 ,0);
        double &pip1 = q(i, j, k + 1, 0);
        double pR = pi - 0.5 * phiR * (pip1 - pi);
        double pL = pim1 + 0.5 * phiL * (pi - pim1);

        double &ci = qh(i, j, k, 3);
        double &cim1 = qh(i, j, k - 1, 3);
        // double &cim2 = qh(i ,j ,k-2 ,3);
        double &cip1 = qh(i, j, k + 1, 3);
        double cR = ci - 0.5 * phiR * (cip1 - ci);
        double cL = cim1 + 0.5 * phiL * (ci - cim1);

        // Compute kinetic energy, total energy
        double kR = 0.5 * (pow(ufR, 2.0) + pow(vfR, 2.0) + pow(wfR, 2.0));
        double kL = 0.5 * (pow(ufL, 2.0) + pow(vfL, 2.0) + pow(wfL, 2.0));
        double ER = rhoR * (eR + kR);
        double EL = rhoL * (eL + kL);

        // Now compute rusanov flux
        // wave speed estimate
        double lam = fmax(abs(UL) + cL, abs(UR) + cR) * S;
        UR *= S;
        UL *= S;

        // Continuity rho*Ui
        double FrhoR, FrhoL;
        FrhoR = UR * rhoR;
        FrhoL = UL * rhoL;
        kF(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

        double FUR, FUL;
        // x momentum rho*u*Ui+ p*Ax
        FUR = UR * ufR * rhoR + pR * kS(i, j, k, 0);
        FUL = UL * ufL * rhoL + pL * kS(i, j, k, 0);
        kF(i, j, k, 1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

        // y momentum rho*v*Ui+ p*Ay
        FUR = UR * vfR * rhoR + pR * kS(i, j, k, 1);
        FUL = UL * vfL * rhoL + pL * kS(i, j, k, 1);
        kF(i, j, k, 2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

        // w momentum rho*w*Ui+ p*Az
        FUR = UR * wfR * rhoR + pR * kS(i, j, k, 2);
        FUL = UL * wfL * rhoL + pL * kS(i, j, k, 2);
        kF(i, j, k, 3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

        // Total energy (rhoE+ p)*Ui)
        double FER, FEL;
        FER = UR * (ER + pR);
        FEL = UL * (EL + pL);
        kF(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

        // Species
        for (int n = 0; n < ne - 5; n++) {
          double FYiR, FYiL;
          double rhoYiR, rhoYiL;
          // Reconstruct Y
          double &rhoYi = Q(i, j, k, 5 + n);
          double &rhoYim1 = Q(i, j, k - 1, 5 + n);
          double &rhoYip1 = Q(i, j, k + 1, 5 + n);
          rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
          rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
          FYiR = rhoYiR * UR;
          FYiL = rhoYiL * UL;
          kF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
        }
      });
}

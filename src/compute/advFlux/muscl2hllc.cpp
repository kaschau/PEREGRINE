#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

// Compute the flux at a face using 2nd order MUSCL reconstruction with HLLC
// flux at face
//
// Here is the strategy:
//
//      Reconstruct   rho, u,v,w e(internal), and Yi
//      Use the reconstruction of e for p and c (speed of sound)
//
//      Use reconstruction of u,v,w to compute kinetic energy at face,
//      then use KE and e to compute total energy at face.
//
//      For the slope limiter, we use the generalized minmod limiter
//      where we can adjust theta E[1,2] where theta=1 is most dissipative
//      and theta=2 is least dissipative (according to wikipedia)

PG_ABI void pgMuscl2hllc(int count, const pgView *Q_, const pgView *iF_,
                         const pgView *iS_, const pgView *jF_,
                         const pgView *jS_, const pgView *kF_,
                         const pgView *kS_, const pgView *q_, const pgView *qh_,
                         const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto iF = as4(iF_[e]);
    auto iS = as4(iS_[e]);
    auto jF = as4(jF_[e]);
    auto jS = as4(jS_[e]);
    auto kF = as4(kF_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;

    double theta = 2.0;
    //-------------------------------------------------------------------------------------------|
    // i flux face range
    //-------------------------------------------------------------------------------------------|
    MDRange3 range_i({ng, ng, ng}, {ni + ng, nj + ng - 1, nk + ng - 1});
    Kokkos::parallel_for(
        "MUSCL 2 hllc i face conv fluxes", range_i,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          double S, nx, ny, nz;
          faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                     nz);

          double rR, rL, phiR, phiL;

          // Reconstruct density
          double &rhoi = Q(i, j, k, 0);
          double &rhoim1 = Q(i - 1, j, k, 0);
          double &rhoim2 = Q(i - 2, j, k, 0);
          double &rhoip1 = Q(i + 1, j, k, 0);
          rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
          rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
          phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
          phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

          double rhoR = rhoi - 0.5 * phiR * (rhoip1 - rhoi);
          double rhoL = rhoim1 + 0.5 * phiL * (rhoi - rhoim1);

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

          double UR = nx * ufR + ny * vfR + nz * wfR;
          double UL = nx * ufL + ny * vfL + nz * wfL;

          // Compute momentums
          double rhouR = rhoR * ufR;
          double rhouL = rhoL * ufL;
          double rhovR = rhoR * vfR;
          double rhovL = rhoL * vfL;
          double rhowR = rhoR * wfR;
          double rhowL = rhoL * wfL;

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

          // Now compute HLLC flux
          double pstar = 0.5 * (pL + pR) - 0.5 * (UR - UL) * 0.5 *
                                               (rhoL + rhoR) * 0.5 * (cL + cR);
          pstar = fmax(0.0, pstar);

          // wave speed estimate
          double SL = UL - cL;
          double SR = UR + cR;
          double Sstar =
              (pR - pL + rhoL * UL * (SL - UL) - rhoR * UR * (SR - UR)) /
              (rhoL * (SL - UL) - rhoR * (SR - UR));

          if (SL >= 0.0) {
            iF(i, j, k, 0) = UL * rhoL * S;
            iF(i, j, k, 1) = UL * rhouL * S + pL * iS(i, j, k, 0);
            iF(i, j, k, 2) = UL * rhovL * S + pL * iS(i, j, k, 1);
            iF(i, j, k, 3) = UL * rhowL * S + pL * iS(i, j, k, 2);
            iF(i, j, k, 4) = UL * (EL + pL) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i - 1, j, k, 5 + n);
              double &Yim2 = q(i - 2, j, k, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              iF(i, j, k, 5 + n) = UL * rhoYL * S;
            }
          } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
            double FrhoL, FUL, FVL, FWL, FEL, UstarL;
            FrhoL = UL * rhoL * S;
            FUL = UL * rhouL * S + pL * iS(i, j, k, 0);
            FVL = UL * rhovL * S + pL * iS(i, j, k, 1);
            FWL = UL * rhowL * S + pL * iS(i, j, k, 2);
            FEL = UL * (EL + pL) * S;
            UstarL = rhoL * (SL - UL) / (SL - Sstar);

            iF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * S;
            iF(i, j, k, 1) = FUL + SL * (UstarL * Sstar * nx - rhouL) * S;
            iF(i, j, k, 2) = FVL + SL * (UstarL * Sstar * ny - rhovL) * S;
            iF(i, j, k, 3) = FWL + SL * (UstarL * Sstar * nz - rhowL) * S;
            iF(i, j, k, 4) =
                FEL + SL *
                          (UstarL * (EL / rhoL +
                                     (Sstar - UL) *
                                         (Sstar + pL / (rhoL * (SL - UL)))) -
                           EL) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i - 1, j, k, 5 + n);
              double &Yim2 = q(i - 2, j, k, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              double FYL = rhoYL * UL * S;
              iF(i, j, k, 5 + n) = FYL + SL * (UstarL * YL - rhoYL) * S;
            }
          } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
            double FrhoR, FUR, FVR, FWR, FER, UstarR;
            FrhoR = UR * rhoR * S;
            FUR = UR * rhouR * S + pR * iS(i, j, k, 0);
            FVR = UR * rhovR * S + pR * iS(i, j, k, 1);
            FWR = UR * rhowR * S + pR * iS(i, j, k, 2);
            FER = UR * (ER + pR) * S;
            UstarR = rhoR * (SR - UR) / (SR - Sstar);

            iF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * S;
            iF(i, j, k, 1) = FUR + SR * (UstarR * Sstar * nx - rhouR) * S;
            iF(i, j, k, 2) = FVR + SR * (UstarR * Sstar * ny - rhovR) * S;
            iF(i, j, k, 3) = FWR + SR * (UstarR * Sstar * nz - rhowR) * S;
            iF(i, j, k, 4) =
                FER + SR *
                          (UstarR * (ER / rhoR +
                                     (Sstar - UR) *
                                         (Sstar + pR / (rhoR * (SR - UR)))) -
                           ER) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i - 1, j, k, 5 + n);
              double &Yip1 = q(i + 1, j, k, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              double FYR = rhoYR * UR * S;
              iF(i, j, k, 5 + n) = FYR + SR * (UstarR * YR - rhoYR) * S;
            }
          } else if (SR <= 0.0) {
            iF(i, j, k, 0) = UR * rhoR * S;
            iF(i, j, k, 1) = UR * rhouR * S + pR * iS(i, j, k, 0);
            iF(i, j, k, 2) = UR * rhovR * S + pR * iS(i, j, k, 1);
            iF(i, j, k, 3) = UR * rhowR * S + pR * iS(i, j, k, 2);
            iF(i, j, k, 4) = UR * (ER + pR) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i - 1, j, k, 5 + n);
              double &Yip1 = q(i + 1, j, k, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              iF(i, j, k, 5 + n) = UR * rhoYR * S;
            }
          }
        });

    //-------------------------------------------------------------------------------------------|
    // j flux face range
    //-------------------------------------------------------------------------------------------|
    MDRange3 range_j({ng, ng, ng}, {ni + ng - 1, nj + ng, nk + ng - 1});
    Kokkos::parallel_for(
        "MUSCL 2 hllc j face conv fluxes", range_j,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          double S, nx, ny, nz;
          faceNormal(jS(i, j, k, 0), jS(i, j, k, 1), jS(i, j, k, 2), S, nx, ny,
                     nz);

          double rR, rL, phiR, phiL;

          // Reconstruct density
          double &rhoi = Q(i, j, k, 0);
          double &rhoim1 = Q(i, j - 1, k, 0);
          double &rhoim2 = Q(i, j - 2, k, 0);
          double &rhoip1 = Q(i, j + 1, k, 0);
          rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
          rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
          phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
          phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

          double rhoR = rhoi - 0.5 * phiR * (rhoip1 - rhoi);
          double rhoL = rhoim1 + 0.5 * phiL * (rhoi - rhoim1);

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

          double UR = nx * ufR + ny * vfR + nz * wfR;
          double UL = nx * ufL + ny * vfL + nz * wfL;

          // Compute momentums
          double rhouR = rhoR * ufR;
          double rhouL = rhoL * ufL;
          double rhovR = rhoR * vfR;
          double rhovL = rhoL * vfL;
          double rhowR = rhoR * wfR;
          double rhowL = rhoL * wfL;

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

          // Now compute HLLC flux
          double pstar = 0.5 * (pL + pR) - 0.5 * (UR - UL) * 0.5 *
                                               (rhoL + rhoR) * 0.5 * (cL + cR);
          pstar = fmax(0.0, pstar);

          // wave speed estimate
          double SL = UL - cL;
          double SR = UR + cR;
          double Sstar =
              (pR - pL + rhoL * UL * (SL - UL) - rhoR * UR * (SR - UR)) /
              (rhoL * (SL - UL) - rhoR * (SR - UR));

          if (SL >= 0.0) {
            jF(i, j, k, 0) = UL * rhoL * S;
            jF(i, j, k, 1) = UL * rhouL * S + pL * jS(i, j, k, 0);
            jF(i, j, k, 2) = UL * rhovL * S + pL * jS(i, j, k, 1);
            jF(i, j, k, 3) = UL * rhowL * S + pL * jS(i, j, k, 2);
            jF(i, j, k, 4) = UL * (EL + pL) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j - 1, k, 5 + n);
              double &Yim2 = q(i, j - 2, k, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              jF(i, j, k, 5 + n) = UL * rhoYL * S;
            }
          } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
            double FrhoL, FUL, FVL, FWL, FEL, UstarL;
            FrhoL = UL * rhoL * S;
            FUL = UL * rhouL * S + pL * jS(i, j, k, 0);
            FVL = UL * rhovL * S + pL * jS(i, j, k, 1);
            FWL = UL * rhowL * S + pL * jS(i, j, k, 2);
            FEL = UL * (EL + pL) * S;
            UstarL = rhoL * (SL - UL) / (SL - Sstar);

            jF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * S;
            jF(i, j, k, 1) = FUL + SL * (UstarL * Sstar * nx - rhouL) * S;
            jF(i, j, k, 2) = FVL + SL * (UstarL * Sstar * ny - rhovL) * S;
            jF(i, j, k, 3) = FWL + SL * (UstarL * Sstar * nz - rhowL) * S;
            jF(i, j, k, 4) =
                FEL + SL *
                          (UstarL * (EL / rhoL +
                                     (Sstar - UL) *
                                         (Sstar + pL / (rhoL * (SL - UL)))) -
                           EL) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j - 1, k, 5 + n);
              double &Yim2 = q(i, j - 2, k, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              double FYL = rhoYL * UL * S;
              jF(i, j, k, 5 + n) = FYL + SL * (UstarL * YL - rhoYL) * S;
            }
          } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
            double FrhoR, FUR, FVR, FWR, FER, UstarR;
            FrhoR = UR * rhoR * S;
            FUR = UR * rhouR * S + pR * jS(i, j, k, 0);
            FVR = UR * rhovR * S + pR * jS(i, j, k, 1);
            FWR = UR * rhowR * S + pR * jS(i, j, k, 2);
            FER = UR * (ER + pR) * S;
            UstarR = rhoR * (SR - UR) / (SR - Sstar);

            jF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * S;
            jF(i, j, k, 1) = FUR + SR * (UstarR * Sstar * nx - rhouR) * S;
            jF(i, j, k, 2) = FVR + SR * (UstarR * Sstar * ny - rhovR) * S;
            jF(i, j, k, 3) = FWR + SR * (UstarR * Sstar * nz - rhowR) * S;
            jF(i, j, k, 4) =
                FER + SR *
                          (UstarR * (ER / rhoR +
                                     (Sstar - UR) *
                                         (Sstar + pR / (rhoR * (SR - UR)))) -
                           ER) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j - 1, k, 5 + n);
              double &Yip1 = q(i, j + 1, k, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              double FYR = rhoYR * UR * S;
              jF(i, j, k, 5 + n) = FYR + SR * (UstarR * YR - rhoYR) * S;
            }
          } else if (SR <= 0.0) {
            jF(i, j, k, 0) = UR * rhoR * S;
            jF(i, j, k, 1) = UR * rhouR * S + pR * jS(i, j, k, 0);
            jF(i, j, k, 2) = UR * rhovR * S + pR * jS(i, j, k, 1);
            jF(i, j, k, 3) = UR * rhowR * S + pR * jS(i, j, k, 2);
            jF(i, j, k, 4) = UR * (ER + pR) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j - 1, k, 5 + n);
              double &Yip1 = q(i, j + 1, k, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              jF(i, j, k, 5 + n) = UR * rhoYR * S;
            }
          }
        });
    //-------------------------------------------------------------------------------------------|
    // k flux face range
    //-------------------------------------------------------------------------------------------|
    MDRange3 range_k({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng});
    Kokkos::parallel_for(
        "MUSCL 2 hllc k face conv fluxes", range_k,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          double S, nx, ny, nz;
          faceNormal(kS(i, j, k, 0), kS(i, j, k, 1), kS(i, j, k, 2), S, nx, ny,
                     nz);

          double rR, rL, phiR, phiL;

          // Reconstruct density
          double &rhoi = Q(i, j, k, 0);
          double &rhoim1 = Q(i, j, k - 1, 0);
          double &rhoim2 = Q(i, j, k - 2, 0);
          double &rhoip1 = Q(i, j, k + 1, 0);
          rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
          rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
          phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
          phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

          double rhoR = rhoi - 0.5 * phiR * (rhoip1 - rhoi);
          double rhoL = rhoim1 + 0.5 * phiL * (rhoi - rhoim1);

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

          double UR = nx * ufR + ny * vfR + nz * wfR;
          double UL = nx * ufL + ny * vfL + nz * wfL;

          // Compute momentums
          double rhouR = rhoR * ufR;
          double rhouL = rhoL * ufL;
          double rhovR = rhoR * vfR;
          double rhovL = rhoL * vfL;
          double rhowR = rhoR * wfR;
          double rhowL = rhoL * wfL;

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

          // Now compute HLLC flux
          double pstar = 0.5 * (pL + pR) - 0.5 * (UR - UL) * 0.5 *
                                               (rhoL + rhoR) * 0.5 * (cL + cR);
          pstar = fmax(0.0, pstar);

          // wave speed estimate
          double SL = UL - cL;
          double SR = UR + cR;
          double Sstar =
              (pR - pL + rhoL * UL * (SL - UL) - rhoR * UR * (SR - UR)) /
              (rhoL * (SL - UL) - rhoR * (SR - UR));

          if (SL >= 0.0) {
            kF(i, j, k, 0) = UL * rhoL * S;
            kF(i, j, k, 1) = UL * rhouL * S + pL * kS(i, j, k, 0);
            kF(i, j, k, 2) = UL * rhovL * S + pL * kS(i, j, k, 1);
            kF(i, j, k, 3) = UL * rhowL * S + pL * kS(i, j, k, 2);
            kF(i, j, k, 4) = UL * (EL + pL) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j, k - 1, 5 + n);
              double &Yim2 = q(i, j, k - 2, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              kF(i, j, k, 5 + n) = UL * rhoYL * S;
            }
          } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
            double FrhoL, FUL, FVL, FWL, FEL, UstarL;
            FrhoL = UL * rhoL * S;
            FUL = UL * rhouL * S + pL * kS(i, j, k, 0);
            FVL = UL * rhovL * S + pL * kS(i, j, k, 1);
            FWL = UL * rhowL * S + pL * kS(i, j, k, 2);
            FEL = UL * (EL + pL) * S;
            UstarL = rhoL * (SL - UL) / (SL - Sstar);

            kF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * S;
            kF(i, j, k, 1) = FUL + SL * (UstarL * Sstar * nx - rhouL) * S;
            kF(i, j, k, 2) = FVL + SL * (UstarL * Sstar * ny - rhovL) * S;
            kF(i, j, k, 3) = FWL + SL * (UstarL * Sstar * nz - rhowL) * S;
            kF(i, j, k, 4) =
                FEL + SL *
                          (UstarL * (EL / rhoL +
                                     (Sstar - UL) *
                                         (Sstar + pL / (rhoL * (SL - UL)))) -
                           EL) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j, k - 1, 5 + n);
              double &Yim2 = q(i, j, k - 2, 5 + n);
              rL = (Yim1 - Yim2) / (Yi - Yim1);
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double YL = Yim1 + 0.5 * phiL * (Yi - Yim1);
              double rhoYL = rhoL * YL;
              double FYL = rhoYL * UL * S;
              kF(i, j, k, 5 + n) = FYL + SL * (UstarL * YL - rhoYL) * S;
            }
          } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
            double FrhoR, FUR, FVR, FWR, FER, UstarR;
            FrhoR = UR * rhoR * S;
            FUR = UR * rhouR * S + pR * kS(i, j, k, 0);
            FVR = UR * rhovR * S + pR * kS(i, j, k, 1);
            FWR = UR * rhowR * S + pR * kS(i, j, k, 2);
            FER = UR * (ER + pR) * S;
            UstarR = rhoR * (SR - UR) / (SR - Sstar);

            kF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * S;
            kF(i, j, k, 1) = FUR + SR * (UstarR * Sstar * nx - rhouR) * S;
            kF(i, j, k, 2) = FVR + SR * (UstarR * Sstar * ny - rhovR) * S;
            kF(i, j, k, 3) = FWR + SR * (UstarR * Sstar * nz - rhowR) * S;
            kF(i, j, k, 4) =
                FER + SR *
                          (UstarR * (ER / rhoR +
                                     (Sstar - UR) *
                                         (Sstar + pR / (rhoR * (SR - UR)))) -
                           ER) *
                          S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j, k - 1, 5 + n);
              double &Yip1 = q(i, j, k + 1, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              double FYR = rhoYR * UR * S;
              kF(i, j, k, 5 + n) = FYR + SR * (UstarR * YR - rhoYR) * S;
            }
          } else if (SR <= 0.0) {
            kF(i, j, k, 0) = UR * rhoR * S;
            kF(i, j, k, 1) = UR * rhouR * S + pR * kS(i, j, k, 0);
            kF(i, j, k, 2) = UR * rhovR * S + pR * kS(i, j, k, 1);
            kF(i, j, k, 3) = UR * rhowR * S + pR * kS(i, j, k, 2);
            kF(i, j, k, 4) = UR * (ER + pR) * S;
            for (int n = 0; n < ne - 5; n++) {
              // Reconstruct Y
              double &Yi = q(i, j, k, 5 + n);
              double &Yim1 = q(i, j, k - 1, 5 + n);
              double &Yip1 = q(i, j, k + 1, 5 + n);
              rR = (Yi - Yim1) / (Yip1 - Yi);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));

              double YR = Yi - 0.5 * phiR * (Yip1 - Yi);
              double rhoYR = rhoR * YR;
              kF(i, j, k, 5 + n) = UR * rhoYR * S;
            }
          }
        });
  }
}

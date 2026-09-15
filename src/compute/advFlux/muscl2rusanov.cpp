#include "kernel.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

PG_STENCIL(2);

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

// compiled once per direction: the jit says which, and the index offsets
// fold into the addressing
#ifndef PG_DIRECTION
#error                                                                         \
    "a flux kernel is compiled for one direction: -DPG_DIRECTION from the jit"
#endif
constexpr int iMod = PG_DIRECTION == 0, jMod = PG_DIRECTION == 1,
              kMod = PG_DIRECTION == 2;

PG_RANGE(faces)
PG_ABI void pgMuscl2rusanov(pgIn *Q_, pgOut *F_, pgIn *A_, pgIn *q_, pgIn *qh_,
                            const pgDims *d, const pgTiling &t) {
#if PG_DIRECTION == 0
  Kokkos::parallel_for(
      "MUSCL 2 rusanov i face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        double theta = 2.0;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double S, nx, ny, nz;
              faceNormal(A(i, j, k, 0), A(i, j, k, 1), A(i, j, k, 2), S, nx, ny,
                         nz);

              double rR, rL, phiR, phiL;

              // Reconstruct density
              // We store density reconstrution values for species
              double phiRhoR, phiRhoL;
              const double &rhoi = Q(i, j, k, 0);
              const double &rhoim1 = Q(i - 1, j, k, 0);
              const double &rhoim2 = Q(i - 2, j, k, 0);
              const double &rhoip1 = Q(i + 1, j, k, 0);
              rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
              rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
              phiRhoR =
                  fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiRhoL =
                  fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
              double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

              // Reconstruct u
              const double &ui = q(i, j, k, 1);
              const double &uim1 = q(i - 1, j, k, 1);
              const double &uim2 = q(i - 2, j, k, 1);
              const double &uip1 = q(i + 1, j, k, 1);
              rR = (ui - uim1) / (uip1 - ui);
              rL = (uim1 - uim2) / (ui - uim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double ufR = ui - 0.5 * phiR * (uip1 - ui);
              double ufL = uim1 + 0.5 * phiL * (ui - uim1);

              // Reconstruct v
              const double &vi = q(i, j, k, 2);
              const double &vim1 = q(i - 1, j, k, 2);
              const double &vim2 = q(i - 2, j, k, 2);
              const double &vip1 = q(i + 1, j, k, 2);
              rR = (vi - vim1) / (vip1 - vi);
              rL = (vim1 - vim2) / (vi - vim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double vfR = vi - 0.5 * phiR * (vip1 - vi);
              double vfL = vim1 + 0.5 * phiL * (vi - vim1);

              // Reconstruct w
              const double &wi = q(i, j, k, 3);
              const double &wim1 = q(i - 1, j, k, 3);
              const double &wim2 = q(i - 2, j, k, 3);
              const double &wip1 = q(i + 1, j, k, 3);
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
              const double &pi = q(i, j, k, 0);
              const double &pim1 = q(i - 1, j, k, 0);
              // const double &pim2 = q(i-2,j ,k ,0);
              const double &pip1 = q(i + 1, j, k, 0);
              double pR = pi - 0.5 * phiR * (pip1 - pi);
              double pL = pim1 + 0.5 * phiL * (pi - pim1);

              const double &ci = qh(i, j, k, 3);
              const double &cim1 = qh(i - 1, j, k, 3);
              // const double &cim2 = qh(i-2,j ,k ,3);
              const double &cip1 = qh(i + 1, j, k, 3);
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
              F(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

              double FUR, FUL;
              // x momentum rho*u*Ui+ p*Ax
              FUR = UR * ufR * rhoR + pR * A(i, j, k, 0);
              FUL = UL * ufL * rhoL + pL * A(i, j, k, 0);
              F(i, j, k, 1) =
                  0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

              // y momentum rho*v*Ui+ p*Ay
              FUR = UR * vfR * rhoR + pR * A(i, j, k, 1);
              FUL = UL * vfL * rhoL + pL * A(i, j, k, 1);
              F(i, j, k, 2) =
                  0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

              // w momentum rho*w*Ui+ p*Az
              FUR = UR * wfR * rhoR + pR * A(i, j, k, 2);
              FUL = UL * wfL * rhoL + pL * A(i, j, k, 2);
              F(i, j, k, 3) =
                  0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

              // Total energy (rhoE+ p)*Ui)
              double FER, FEL;
              FER = UR * (ER + pR);
              FEL = UL * (EL + pL);
              F(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double FYiR, FYiL;
                double rhoYiR, rhoYiL;
                // Reconstruct Y
                const double &rhoYi = Q(i, j, k, 5 + n);
                const double &rhoYim1 = Q(i - 1, j, k, 5 + n);
                const double &rhoYip1 = Q(i + 1, j, k, 5 + n);
                rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
                rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
                FYiR = rhoYiR * UR;
                FYiL = rhoYiL * UL;
                F(i, j, k, 5 + n) =
                    0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
              }
            });
      });
#endif
#if PG_DIRECTION == 1
  Kokkos::parallel_for(
      "MUSCL 2 rusanov j face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        double theta = 2.0;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double S, nx, ny, nz;
              faceNormal(A(i, j, k, 0), A(i, j, k, 1), A(i, j, k, 2), S, nx, ny,
                         nz);

              double rR, rL, phiR, phiL;

              // Reconstruct density
              // We store density reconstrution values for species
              double phiRhoR, phiRhoL;
              const double &rhoi = Q(i, j, k, 0);
              const double &rhoim1 = Q(i, j - 1, k, 0);
              const double &rhoim2 = Q(i, j - 2, k, 0);
              const double &rhoip1 = Q(i, j + 1, k, 0);
              rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
              rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
              phiRhoR =
                  fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiRhoL =
                  fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
              double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

              // Reconstruct u
              const double &ui = q(i, j, k, 1);
              const double &uim1 = q(i, j - 1, k, 1);
              const double &uim2 = q(i, j - 2, k, 1);
              const double &uip1 = q(i, j + 1, k, 1);
              rR = (ui - uim1) / (uip1 - ui);
              rL = (uim1 - uim2) / (ui - uim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double ufR = ui - 0.5 * phiR * (uip1 - ui);
              double ufL = uim1 + 0.5 * phiL * (ui - uim1);

              // Reconstruct v
              const double &vi = q(i, j, k, 2);
              const double &vim1 = q(i, j - 1, k, 2);
              const double &vim2 = q(i, j - 2, k, 2);
              const double &vip1 = q(i, j + 1, k, 2);
              rR = (vi - vim1) / (vip1 - vi);
              rL = (vim1 - vim2) / (vi - vim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double vfR = vi - 0.5 * phiR * (vip1 - vi);
              double vfL = vim1 + 0.5 * phiL * (vi - vim1);

              // Reconstruct w
              const double &wi = q(i, j, k, 3);
              const double &wim1 = q(i, j - 1, k, 3);
              const double &wim2 = q(i, j - 2, k, 3);
              const double &wip1 = q(i, j + 1, k, 3);
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
              const double &pi = q(i, j, k, 0);
              const double &pim1 = q(i, j - 1, k, 0);
              // const double &pim2 = q(i ,j-2 ,k ,0);
              const double &pip1 = q(i, j + 1, k, 0);
              double pR = pi - 0.5 * phiR * (pip1 - pi);
              double pL = pim1 + 0.5 * phiL * (pi - pim1);

              const double &ci = qh(i, j, k, 3);
              const double &cim1 = qh(i, j - 1, k, 3);
              // const double &cim2 = qh(i ,j-2 ,k ,3);
              const double &cip1 = qh(i, j + 1, k, 3);
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
              F(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

              double FUR, FUL;
              // x momentum rho*u*Ui+ p*Ax
              FUR = UR * ufR * rhoR + pR * A(i, j, k, 0);
              FUL = UL * ufL * rhoL + pL * A(i, j, k, 0);
              F(i, j, k, 1) =
                  0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

              // y momentum rho*v*Ui+ p*Ay
              FUR = UR * vfR * rhoR + pR * A(i, j, k, 1);
              FUL = UL * vfL * rhoL + pL * A(i, j, k, 1);
              F(i, j, k, 2) =
                  0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

              // w momentum rho*w*Ui+ p*Az
              FUR = UR * wfR * rhoR + pR * A(i, j, k, 2);
              FUL = UL * wfL * rhoL + pL * A(i, j, k, 2);
              F(i, j, k, 3) =
                  0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

              // Total energy (rhoE+ p)*Ui)
              double FER, FEL;
              FER = UR * (ER + pR);
              FEL = UL * (EL + pL);
              F(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double FYiR, FYiL;
                double rhoYiR, rhoYiL;
                // Reconstruct Y
                const double &rhoYi = Q(i, j, k, 5 + n);
                const double &rhoYim1 = Q(i, j - 1, k, 5 + n);
                const double &rhoYip1 = Q(i, j + 1, k, 5 + n);
                rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
                rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
                FYiR = rhoYiR * UR;
                FYiL = rhoYiL * UL;
                F(i, j, k, 5 + n) =
                    0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
              }
            });
      });
#endif
#if PG_DIRECTION == 2
  Kokkos::parallel_for(
      "MUSCL 2 rusanov k face conv fluxes", t.policy(),
      KOKKOS_LAMBDA(const team &team) {
        const auto r = t.of(team);
        const int e = r.e;
        auto Q = as4(Q_[e]);
        auto F = as4(F_[e]);
        auto A = as4(A_[e]);
        auto q = as4(q_[e]);
        auto qh = as4(qh_[e]);
        const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
        double theta = 2.0;
        //-------------------------------------------------------------------------------------------|
        // i flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // j flux face range
        //-------------------------------------------------------------------------------------------|
        //-------------------------------------------------------------------------------------------|
        // k flux face range
        //-------------------------------------------------------------------------------------------|
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, r.begin, r.end), [&](const int item) {
              int i, j, k;
              cellAt(t.cells[e], item, i, j, k);

              double S, nx, ny, nz;
              faceNormal(A(i, j, k, 0), A(i, j, k, 1), A(i, j, k, 2), S, nx, ny,
                         nz);

              double rR, rL, phiR, phiL;

              // Reconstruct density
              // We store density reconstrution values for species
              double phiRhoR, phiRhoL;
              const double &rhoi = Q(i, j, k, 0);
              const double &rhoim1 = Q(i, j, k - 1, 0);
              const double &rhoim2 = Q(i, j, k - 2, 0);
              const double &rhoip1 = Q(i, j, k + 1, 0);
              rR = (rhoi - rhoim1) / (rhoip1 - rhoi);
              rL = (rhoim1 - rhoim2) / (rhoi - rhoim1);
              phiRhoR =
                  fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiRhoL =
                  fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double rhoR = rhoi - 0.5 * phiRhoR * (rhoip1 - rhoi);
              double rhoL = rhoim1 + 0.5 * phiRhoL * (rhoi - rhoim1);

              // Reconstruct u
              const double &ui = q(i, j, k, 1);
              const double &uim1 = q(i, j, k - 1, 1);
              const double &uim2 = q(i, j, k - 2, 1);
              const double &uip1 = q(i, j, k + 1, 1);
              rR = (ui - uim1) / (uip1 - ui);
              rL = (uim1 - uim2) / (ui - uim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double ufR = ui - 0.5 * phiR * (uip1 - ui);
              double ufL = uim1 + 0.5 * phiL * (ui - uim1);

              // Reconstruct v
              const double &vi = q(i, j, k, 2);
              const double &vim1 = q(i, j, k - 1, 2);
              const double &vim2 = q(i, j, k - 2, 2);
              const double &vip1 = q(i, j, k + 1, 2);
              rR = (vi - vim1) / (vip1 - vi);
              rL = (vim1 - vim2) / (vi - vim1);
              phiR = fmax(0.0, fmin(fmin(theta * rR, (1.0 + rR) / 2.0), theta));
              phiL = fmax(0.0, fmin(fmin(theta * rL, (1.0 + rL) / 2.0), theta));

              double vfR = vi - 0.5 * phiR * (vip1 - vi);
              double vfL = vim1 + 0.5 * phiL * (vi - vim1);

              // Reconstruct w
              const double &wi = q(i, j, k, 3);
              const double &wim1 = q(i, j, k - 1, 3);
              const double &wim2 = q(i, j, k - 2, 3);
              const double &wip1 = q(i, j, k + 1, 3);
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
              const double &pi = q(i, j, k, 0);
              const double &pim1 = q(i, j, k - 1, 0);
              // const double &pim2 = q(i ,j ,k-2 ,0);
              const double &pip1 = q(i, j, k + 1, 0);
              double pR = pi - 0.5 * phiR * (pip1 - pi);
              double pL = pim1 + 0.5 * phiL * (pi - pim1);

              const double &ci = qh(i, j, k, 3);
              const double &cim1 = qh(i, j, k - 1, 3);
              // const double &cim2 = qh(i ,j ,k-2 ,3);
              const double &cip1 = qh(i, j, k + 1, 3);
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
              F(i, j, k, 0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

              double FUR, FUL;
              // x momentum rho*u*Ui+ p*Ax
              FUR = UR * ufR * rhoR + pR * A(i, j, k, 0);
              FUL = UL * ufL * rhoL + pL * A(i, j, k, 0);
              F(i, j, k, 1) =
                  0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

              // y momentum rho*v*Ui+ p*Ay
              FUR = UR * vfR * rhoR + pR * A(i, j, k, 1);
              FUL = UL * vfL * rhoL + pL * A(i, j, k, 1);
              F(i, j, k, 2) =
                  0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

              // w momentum rho*w*Ui+ p*Az
              FUR = UR * wfR * rhoR + pR * A(i, j, k, 2);
              FUL = UL * wfL * rhoL + pL * A(i, j, k, 2);
              F(i, j, k, 3) =
                  0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

              // Total energy (rhoE+ p)*Ui)
              double FER, FEL;
              FER = UR * (ER + pR);
              FEL = UL * (EL + pL);
              F(i, j, k, 4) = 0.5 * (FER + FEL - lam * (ER - EL));

              // Species
              for (int n = 0; n < ne - 5; n++) {
                double FYiR, FYiL;
                double rhoYiR, rhoYiL;
                // Reconstruct Y
                const double &rhoYi = Q(i, j, k, 5 + n);
                const double &rhoYim1 = Q(i, j, k - 1, 5 + n);
                const double &rhoYip1 = Q(i, j, k + 1, 5 + n);
                rhoYiR = rhoYi - 0.5 * phiRhoR * (rhoYip1 - rhoYi);
                rhoYiL = rhoYim1 + 0.5 * phiRhoL * (rhoYi - rhoYim1);
                FYiR = rhoYiR * UR;
                FYiL = rhoYiL * UL;
                F(i, j, k, 5 + n) =
                    0.5 * (FYiR + FYiL - lam * (rhoYiR - rhoYiL));
              }
            });
      });
#endif
}

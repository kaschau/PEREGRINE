#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const fourDview &iS,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "hllc face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double &vfL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double &wfL = b.q(i - iMod, j - jMod, k - kMod, 3);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i - iMod, j - jMod, k - kMod, 0);

        double &rhouR = b.Q(i, j, k, 1);
        double &rhouL = b.Q(i - iMod, j - jMod, k - kMod, 1);
        double &rhovR = b.Q(i, j, k, 2);
        double &rhovL = b.Q(i - iMod, j - jMod, k - kMod, 2);
        double &rhowR = b.Q(i, j, k, 3);
        double &rhowL = b.Q(i - iMod, j - jMod, k - kMod, 3);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i - iMod, j - jMod, k - kMod, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i - iMod, j - jMod, k - kMod, 4);

        double &cR = b.qh(i, j, k, 3);
        double &cL = b.qh(i - iMod, j - jMod, k - kMod, 3);

        double pstar = 0.5 * (pL + pR) -
                       0.5 * (UR - UL) * 0.5 * (rhoL + rhoR) * 0.5 * (cL + cR);
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
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) = UL * rhoYiL * S;
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
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  S;
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n) * UL * S;
            YiL = b.q(i - iMod, j - jMod, k - kMod, 5 + n);
            rhoYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) = FYiL + SL * (UstarL * YiL - rhoYiL) * S;
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
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  S;
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * S;
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) = FYiR + SR * (UstarR * YiR - rhoYiR) * S;
          }
        } else if (SR <= 0.0) {
          iF(i, j, k, 0) = UR * rhoR * S;
          iF(i, j, k, 1) = UR * rhouR * S + pR * iS(i, j, k, 0);
          iF(i, j, k, 2) = UR * rhovR * S + pR * iS(i, j, k, 1);
          iF(i, j, k, 3) = UR * rhowR * S + pR * iS(i, j, k, 2);
          iF(i, j, k, 4) = UR * (ER + pR) * S;
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) = UR * rhoYiR * S;
          }
        }
      });
}

void hllc(block_ &b) {
  computeFlux(b, b.iF, b.iS, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, 0, 0, 1);
}

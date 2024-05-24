#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &iS,
                        const threeDview &isx, const threeDview &isy,
                        const threeDview &isz, const threeDview &inx,
                        const threeDview &iny, const threeDview &inz,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "hllc face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double &vfL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double &wfL = b.q(i - iMod, j - jMod, k - kMod, 3);

        double &nx = inx(i, j, k);
        double &ny = iny(i, j, k);
        double &nz = inz(i, j, k);

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
          iF(i, j, k, 0) = UL * rhoL * iS(i, j, k);
          iF(i, j, k, 1) = UL * rhouL * iS(i, j, k) + pL * isx(i, j, k);
          iF(i, j, k, 2) = UL * rhovL * iS(i, j, k) + pL * isy(i, j, k);
          iF(i, j, k, 3) = UL * rhowL * iS(i, j, k) + pL * isz(i, j, k);
          iF(i, j, k, 4) = UL * (EL + pL) * iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) = UL * rhoYiL * iS(i, j, k);
          }
        } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL, FUL, FVL, FWL, FEL, UstarL;
          FrhoL = UL * rhoL * iS(i, j, k);
          FUL = UL * rhouL * iS(i, j, k) + pL * isx(i, j, k);
          FVL = UL * rhovL * iS(i, j, k) + pL * isy(i, j, k);
          FWL = UL * rhowL * iS(i, j, k) + pL * isz(i, j, k);
          FEL = UL * (EL + pL) * iS(i, j, k);
          UstarL = rhoL * (SL - UL) / (SL - Sstar);

          iF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * iS(i, j, k);
          iF(i, j, k, 1) =
              FUL + SL * (UstarL * Sstar * nx - rhouL) * iS(i, j, k);
          iF(i, j, k, 2) =
              FVL + SL * (UstarL * Sstar * ny - rhovL) * iS(i, j, k);
          iF(i, j, k, 3) =
              FWL + SL * (UstarL * Sstar * nz - rhowL) * iS(i, j, k);
          iF(i, j, k, 4) =
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n) * UL * iS(i, j, k);
            YiL = b.q(i - iMod, j - jMod, k - kMod, 5 + n);
            rhoYiL = b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
            iF(i, j, k, 5 + n) =
                FYiL + SL * (UstarL * YiL - rhoYiL) * iS(i, j, k);
          }
        } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR, FUR, FVR, FWR, FER, UstarR;
          FrhoR = UR * rhoR * iS(i, j, k);
          FUR = UR * rhouR * iS(i, j, k) + pR * isx(i, j, k);
          FVR = UR * rhovR * iS(i, j, k) + pR * isy(i, j, k);
          FWR = UR * rhowR * iS(i, j, k) + pR * isz(i, j, k);
          FER = UR * (ER + pR) * iS(i, j, k);
          UstarR = rhoR * (SR - UR) / (SR - Sstar);

          iF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * iS(i, j, k);
          iF(i, j, k, 1) =
              FUR + SR * (UstarR * Sstar * nx - rhouR) * iS(i, j, k);
          iF(i, j, k, 2) =
              FVR + SR * (UstarR * Sstar * ny - rhovR) * iS(i, j, k);
          iF(i, j, k, 3) =
              FWR + SR * (UstarR * Sstar * nz - rhowR) * iS(i, j, k);
          iF(i, j, k, 4) =
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * iS(i, j, k);
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) =
                FYiR + SR * (UstarR * YiR - rhoYiR) * iS(i, j, k);
          }
        } else if (SR <= 0.0) {
          iF(i, j, k, 0) = UR * rhoR * iS(i, j, k);
          iF(i, j, k, 1) = UR * rhouR * iS(i, j, k) + pR * isx(i, j, k);
          iF(i, j, k, 2) = UR * rhovR * iS(i, j, k) + pR * isy(i, j, k);
          iF(i, j, k, 3) = UR * rhowR * iS(i, j, k) + pR * isz(i, j, k);
          iF(i, j, k, 4) = UR * (ER + pR) * iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            iF(i, j, k, 5 + n) = UR * rhoYiR * iS(i, j, k);
          }
        }
      });
}

void hllc(block_ &b) {
  computeFlux(b, b.iF, b.iS, b.isx, b.isy, b.isz, b.inx, b.iny, b.inz, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, b.jsx, b.jsy, b.jsz, b.jnx, b.jny, b.jnz, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, b.ksx, b.ksy, b.ksz, b.knx, b.kny, b.knz, 0, 0, 1);
}

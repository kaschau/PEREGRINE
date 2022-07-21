#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"

void hllc(block_ b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "hllc i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - 1, j, k, 1);
        double &vfL = b.q(i - 1, j, k, 2);
        double &wfL = b.q(i - 1, j, k, 3);

        double &nx = b.inx(i, j, k);
        double &ny = b.iny(i, j, k);
        double &nz = b.inz(i, j, k);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

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
          b.iF(i, j, k, 0) = UL * rhoL * b.iS(i, j, k);
          b.iF(i, j, k, 1) = UL * rhouL * b.iS(i, j, k) + pL * b.isx(i, j, k);
          b.iF(i, j, k, 2) = UL * rhovL * b.iS(i, j, k) + pL * b.isy(i, j, k);
          b.iF(i, j, k, 3) = UL * rhowL * b.iS(i, j, k) + pL * b.isz(i, j, k);
          b.iF(i, j, k, 4) = UL * (EL + pL) * b.iS(i, j, k);
          for (int n = 0; n < b.ne-5; n++) {
            double rhoYiL = b.Q(i - 1, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = UL * rhoYiL * b.iS(i, j, k);
          }
        } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL, FUL, FVL, FWL, FEL, UstarL;
          FrhoL = UL * rhoL * b.iS(i, j, k);
          FUL = UL * rhouL * b.iS(i, j, k) + pL * b.isx(i, j, k);
          FVL = UL * rhovL * b.iS(i, j, k) + pL * b.isy(i, j, k);
          FWL = UL * rhowL * b.iS(i, j, k) + pL * b.isz(i, j, k);
          FEL = UL * (EL + pL) * b.iS(i, j, k);
          UstarL = rhoL * (SL - UL) / (SL - Sstar);

          b.iF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * b.iS(i, j, k);
          b.iF(i, j, k, 1) =
              FUL + SL * (UstarL * Sstar * nx - rhouL) * b.iS(i, j, k);
          b.iF(i, j, k, 2) =
              FVL + SL * (UstarL * Sstar * ny - rhovL) * b.iS(i, j, k);
          b.iF(i, j, k, 3) =
              FWL + SL * (UstarL * Sstar * nz - rhowL) * b.iS(i, j, k);
          b.iF(i, j, k, 4) =
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  b.iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i - 1, j, k, 5 + n) * UL * b.iS(i, j, k);
            YiL = b.q(i - 1, j, k, 5 + n);
            rhoYiL = b.Q(i - 1, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) =
                FYiL + SL * (UstarL * YiL - rhoYiL) * b.iS(i, j, k);
          }
        } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR, FUR, FVR, FWR, FER, UstarR;
          FrhoR = UR * rhoR * b.iS(i, j, k);
          FUR = UR * rhouR * b.iS(i, j, k) + pR * b.isx(i, j, k);
          FVR = UR * rhovR * b.iS(i, j, k) + pR * b.isy(i, j, k);
          FWR = UR * rhowR * b.iS(i, j, k) + pR * b.isz(i, j, k);
          FER = UR * (ER + pR) * b.iS(i, j, k);
          UstarR = rhoR * (SR - UR) / (SR - Sstar);

          b.iF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * b.iS(i, j, k);
          b.iF(i, j, k, 1) =
              FUR + SR * (UstarR * Sstar * nx - rhouR) * b.iS(i, j, k);
          b.iF(i, j, k, 2) =
              FVR + SR * (UstarR * Sstar * ny - rhovR) * b.iS(i, j, k);
          b.iF(i, j, k, 3) =
              FWR + SR * (UstarR * Sstar * nz - rhowR) * b.iS(i, j, k);
          b.iF(i, j, k, 4) =
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  b.iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * b.iS(i, j, k);
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) =
                FYiR + SR * (UstarR * YiR - rhoYiR) * b.iS(i, j, k);
          }
        } else if (SR <= 0.0) {
          b.iF(i, j, k, 0) = UR * rhoR * b.iS(i, j, k);
          b.iF(i, j, k, 1) = UR * rhouR * b.iS(i, j, k) + pR * b.isx(i, j, k);
          b.iF(i, j, k, 2) = UR * rhovR * b.iS(i, j, k) + pR * b.isy(i, j, k);
          b.iF(i, j, k, 3) = UR * rhowR * b.iS(i, j, k) + pR * b.isz(i, j, k);
          b.iF(i, j, k, 4) = UR * (ER + pR) * b.iS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            b.iF(i, j, k, 5 + n) = UR * rhoYiR * b.iS(i, j, k);
          }
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "hllc j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i, j - 1, k, 1);
        double &vfL = b.q(i, j - 1, k, 2);
        double &wfL = b.q(i, j - 1, k, 3);

        double &nx = b.jnx(i, j, k);
        double &ny = b.jny(i, j, k);
        double &nz = b.jnz(i, j, k);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i, j - 1, k, 0);

        double &rhouR = b.Q(i, j, k, 1);
        double &rhouL = b.Q(i, j - 1, k, 1);
        double &rhovR = b.Q(i, j, k, 2);
        double &rhovL = b.Q(i, j - 1, k, 2);
        double &rhowR = b.Q(i, j, k, 3);
        double &rhowL = b.Q(i, j - 1, k, 3);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i, j - 1, k, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i, j - 1, k, 4);

        double &cR = b.qh(i, j, k, 3);
        double &cL = b.qh(i, j - 1, k, 3);

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
          b.jF(i, j, k, 0) = UL * rhoL * b.jS(i, j, k);
          b.jF(i, j, k, 1) = UL * rhouL * b.jS(i, j, k) + pL * b.jsx(i, j, k);
          b.jF(i, j, k, 2) = UL * rhovL * b.jS(i, j, k) + pL * b.jsy(i, j, k);
          b.jF(i, j, k, 3) = UL * rhowL * b.jS(i, j, k) + pL * b.jsz(i, j, k);
          b.jF(i, j, k, 4) = UL * (EL + pL) * b.jS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiL = b.Q(i, j - 1, k, 5 + n);
            b.jF(i, j, k, 5 + n) = UL * rhoYiL * b.jS(i, j, k);
          }
        } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL, FUL, FVL, FWL, FEL, UstarL;
          FrhoL = UL * rhoL * b.jS(i, j, k);
          FUL = UL * rhouL * b.jS(i, j, k) + pL * b.jsx(i, j, k);
          FVL = UL * rhovL * b.jS(i, j, k) + pL * b.jsy(i, j, k);
          FWL = UL * rhowL * b.jS(i, j, k) + pL * b.jsz(i, j, k);
          FEL = UL * (EL + pL) * b.jS(i, j, k);
          UstarL = rhoL * (SL - UL) / (SL - Sstar);

          b.jF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * b.jS(i, j, k);
          b.jF(i, j, k, 1) =
              FUL + SL * (UstarL * Sstar * nx - rhouL) * b.jS(i, j, k);
          b.jF(i, j, k, 2) =
              FVL + SL * (UstarL * Sstar * ny - rhovL) * b.jS(i, j, k);
          b.jF(i, j, k, 3) =
              FWL + SL * (UstarL * Sstar * nz - rhowL) * b.jS(i, j, k);
          b.jF(i, j, k, 4) =
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  b.jS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i, j - 1, k, 5 + n) * UL * b.jS(i, j, k);
            YiL = b.q(i, j - 1, k, 5 + n);
            rhoYiL = b.Q(i, j - 1, k, 5 + n);
            b.jF(i, j, k, 5 + n) =
                FYiL + SL * (UstarL * YiL - rhoYiL) * b.jS(i, j, k);
          }
        } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR, FUR, FVR, FWR, FER, UstarR;
          FrhoR = UR * rhoR * b.jS(i, j, k);
          FUR = UR * rhouR * b.jS(i, j, k) + pR * b.jsx(i, j, k);
          FVR = UR * rhovR * b.jS(i, j, k) + pR * b.jsy(i, j, k);
          FWR = UR * rhowR * b.jS(i, j, k) + pR * b.jsz(i, j, k);
          FER = UR * (ER + pR) * b.jS(i, j, k);
          UstarR = rhoR * (SR - UR) / (SR - Sstar);

          b.jF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * b.jS(i, j, k);
          b.jF(i, j, k, 1) =
              FUR + SR * (UstarR * Sstar * nx - rhouR) * b.jS(i, j, k);
          b.jF(i, j, k, 2) =
              FVR + SR * (UstarR * Sstar * ny - rhovR) * b.jS(i, j, k);
          b.jF(i, j, k, 3) =
              FWR + SR * (UstarR * Sstar * nz - rhowR) * b.jS(i, j, k);
          b.jF(i, j, k, 4) =
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  b.jS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * b.jS(i, j, k);
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            b.jF(i, j, k, 5 + n) =
                FYiR + SR * (UstarR * YiR - rhoYiR) * b.jS(i, j, k);
          }
        } else if (SR <= 0.0) {
          b.jF(i, j, k, 0) = UR * rhoR * b.jS(i, j, k);
          b.jF(i, j, k, 1) = UR * rhouR * b.jS(i, j, k) + pR * b.jsx(i, j, k);
          b.jF(i, j, k, 2) = UR * rhovR * b.jS(i, j, k) + pR * b.jsy(i, j, k);
          b.jF(i, j, k, 3) = UR * rhowR * b.jS(i, j, k) + pR * b.jsz(i, j, k);
          b.jF(i, j, k, 4) = UR * (ER + pR) * b.jS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            b.jF(i, j, k, 5 + n) = UR * rhoYiR * b.jS(i, j, k);
          }
        }
      });
  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "hllc k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i, j, k - 1, 1);
        double &vfL = b.q(i, j, k - 1, 2);
        double &wfL = b.q(i, j, k - 1, 3);

        double &nx = b.knx(i, j, k);
        double &ny = b.kny(i, j, k);
        double &nz = b.knz(i, j, k);

        double UR = nx * ufR + ny * vfR + nz * wfR;
        double UL = nx * ufL + ny * vfL + nz * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i, j, k - 1, 0);

        double &rhouR = b.Q(i, j, k, 1);
        double &rhouL = b.Q(i, j, k - 1, 1);
        double &rhovR = b.Q(i, j, k, 2);
        double &rhovL = b.Q(i, j, k - 1, 2);
        double &rhowR = b.Q(i, j, k, 3);
        double &rhowL = b.Q(i, j, k - 1, 3);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i, j, k - 1, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i, j, k - 1, 4);

        double &cR = b.qh(i, j, k, 3);
        double &cL = b.qh(i, j, k - 1, 3);

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
          b.kF(i, j, k, 0) = UL * rhoL * b.kS(i, j, k);
          b.kF(i, j, k, 1) = UL * rhouL * b.kS(i, j, k) + pL * b.ksx(i, j, k);
          b.kF(i, j, k, 2) = UL * rhovL * b.kS(i, j, k) + pL * b.ksy(i, j, k);
          b.kF(i, j, k, 3) = UL * rhowL * b.kS(i, j, k) + pL * b.ksz(i, j, k);
          b.kF(i, j, k, 4) = UL * (EL + pL) * b.kS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiL = b.Q(i, j, k - 1, 5 + n);
            b.kF(i, j, k, 5 + n) = UL * rhoYiL * b.kS(i, j, k);
          }
        } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
          double FrhoL, FUL, FVL, FWL, FEL, UstarL;
          FrhoL = UL * rhoL * b.kS(i, j, k);
          FUL = UL * rhouL * b.kS(i, j, k) + pL * b.ksx(i, j, k);
          FVL = UL * rhovL * b.kS(i, j, k) + pL * b.ksy(i, j, k);
          FWL = UL * rhowL * b.kS(i, j, k) + pL * b.ksz(i, j, k);
          FEL = UL * (EL + pL) * b.kS(i, j, k);
          UstarL = rhoL * (SL - UL) / (SL - Sstar);

          b.kF(i, j, k, 0) = FrhoL + SL * (UstarL - rhoL) * b.kS(i, j, k);
          b.kF(i, j, k, 1) =
              FUL + SL * (UstarL * Sstar * nx - rhouL) * b.kS(i, j, k);
          b.kF(i, j, k, 2) =
              FVL + SL * (UstarL * Sstar * ny - rhovL) * b.kS(i, j, k);
          b.kF(i, j, k, 3) =
              FWL + SL * (UstarL * Sstar * nz - rhowL) * b.kS(i, j, k);
          b.kF(i, j, k, 4) =
              FEL +
              SL *
                  (UstarL * (EL / rhoL +
                             (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                   EL) *
                  b.kS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiL, YiL, rhoYiL;
            FYiL = b.Q(i, j, k - 1, 5 + n) * UL * b.kS(i, j, k);
            YiL = b.q(i, j, k - 1, 5 + n);
            rhoYiL = b.Q(i, j, k - 1, 5 + n);
            b.kF(i, j, k, 5 + n) =
                FYiL + SL * (UstarL * YiL - rhoYiL) * b.kS(i, j, k);
          }
        } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
          double FrhoR, FUR, FVR, FWR, FER, UstarR;
          FrhoR = UR * rhoR * b.kS(i, j, k);
          FUR = UR * rhouR * b.kS(i, j, k) + pR * b.ksx(i, j, k);
          FVR = UR * rhovR * b.kS(i, j, k) + pR * b.ksy(i, j, k);
          FWR = UR * rhowR * b.kS(i, j, k) + pR * b.ksz(i, j, k);
          FER = UR * (ER + pR) * b.kS(i, j, k);
          UstarR = rhoR * (SR - UR) / (SR - Sstar);

          b.kF(i, j, k, 0) = FrhoR + SR * (UstarR - rhoR) * b.kS(i, j, k);
          b.kF(i, j, k, 1) =
              FUR + SR * (UstarR * Sstar * nx - rhouR) * b.kS(i, j, k);
          b.kF(i, j, k, 2) =
              FVR + SR * (UstarR * Sstar * ny - rhovR) * b.kS(i, j, k);
          b.kF(i, j, k, 3) =
              FWR + SR * (UstarR * Sstar * nz - rhowR) * b.kS(i, j, k);
          b.kF(i, j, k, 4) =
              FER +
              SR *
                  (UstarR * (ER / rhoR +
                             (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                   ER) *
                  b.kS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double FYiR, YiR, rhoYiR;
            FYiR = b.Q(i, j, k, 5 + n) * UR * b.kS(i, j, k);
            YiR = b.q(i, j, k, 5 + n);
            rhoYiR = b.Q(i, j, k, 5 + n);
            b.kF(i, j, k, 5 + n) =
                FYiR + SR * (UstarR * YiR - rhoYiR) * b.kS(i, j, k);
          }
        } else if (SR <= 0.0) {
          b.kF(i, j, k, 0) = UR * rhoR * b.kS(i, j, k);
          b.kF(i, j, k, 1) = UR * rhouR * b.kS(i, j, k) + pR * b.ksx(i, j, k);
          b.kF(i, j, k, 2) = UR * rhovR * b.kS(i, j, k) + pR * b.ksy(i, j, k);
          b.kF(i, j, k, 3) = UR * rhowR * b.kS(i, j, k) + pR * b.ksz(i, j, k);
          b.kF(i, j, k, 4) = UR * (ER + pR) * b.kS(i, j, k);
          for (int n = 0; n < b.ne - 5; n++) {
            double rhoYiR = b.Q(i, j, k, 5 + n);
            b.kF(i, j, k, 5 + n) = UR * rhoYiR * b.kS(i, j, k);
          }
        }
      });
}

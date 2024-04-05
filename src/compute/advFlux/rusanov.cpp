#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

void rusanov(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "rusanov i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double UR;
        double UL;

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i - 1, j, k, 1);
        double &vfL = b.q(i - 1, j, k, 2);
        double &wfL = b.q(i - 1, j, k, 3);

        UR = b.inx(i, j, k) * ufR + b.iny(i, j, k) * vfR + b.inz(i, j, k) * wfR;
        UL = b.inx(i, j, k) * ufL + b.iny(i, j, k) * vfL + b.inz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i - 1, j, k, 0);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i - 1, j, k, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i - 1, j, k, 4);

        // wave speed estimate
        double lam =
            fmax(abs(UL) + b.qh(i, j, k, 3), abs(UR) + b.qh(i - 1, j, k, 3)) *
            b.iS(i, j, k);
        UR *= b.iS(i, j, k);
        UL *= b.iS(i, j, k);

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
        double FYiR, FYiL;
        double YiR, YiL;
        for (int n = 0; n < b.ne - 5; n++) {
          FYiR = b.Q(i, j, k, 5 + n) * UR;
          FYiL = b.Q(i - 1, j, k, 5 + n) * UL;
          YiR = b.Q(i, j, k, 5 + n);
          YiL = b.Q(i - 1, j, k, 5 + n);
          b.iF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "rusanov j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double UR;
        double UL;

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i, j - 1, k, 1);
        double &vfL = b.q(i, j - 1, k, 2);
        double &wfL = b.q(i, j - 1, k, 3);

        UR = b.jnx(i, j, k) * ufR + b.jny(i, j, k) * vfR + b.jnz(i, j, k) * wfR;
        UL = b.jnx(i, j, k) * ufL + b.jny(i, j, k) * vfL + b.jnz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i, j - 1, k, 0);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i, j - 1, k, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i, j - 1, k, 4);

        // wave speed estimate
        double lam =
            fmax(abs(UL) + b.qh(i, j, k, 3), abs(UR) + b.qh(i, j - 1, k, 3)) *
            b.jS(i, j, k);
        UR *= b.jS(i, j, k);
        UL *= b.jS(i, j, k);

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
        double FYiR, FYiL;
        double YiR, YiL;
        for (int n = 0; n < b.ne - 5; n++) {
          FYiR = b.Q(i, j, k, 5 + n) * UR;
          FYiL = b.Q(i, j - 1, k, 5 + n) * UL;
          YiR = b.Q(i, j, k, 5 + n);
          YiL = b.Q(i, j - 1, k, 5 + n);
          b.jF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
        }
      });
  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "rusanov k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double UR;
        double UL;

        double &ufR = b.q(i, j, k, 1);
        double &vfR = b.q(i, j, k, 2);
        double &wfR = b.q(i, j, k, 3);

        double &ufL = b.q(i, j, k - 1, 1);
        double &vfL = b.q(i, j, k - 1, 2);
        double &wfL = b.q(i, j, k - 1, 3);

        UR = b.knx(i, j, k) * ufR + b.kny(i, j, k) * vfR + b.knz(i, j, k) * wfR;
        UL = b.knx(i, j, k) * ufL + b.kny(i, j, k) * vfL + b.knz(i, j, k) * wfL;

        double &rhoR = b.Q(i, j, k, 0);
        double &rhoL = b.Q(i, j, k - 1, 0);

        double &pR = b.q(i, j, k, 0);
        double &pL = b.q(i, j, k - 1, 0);

        double &ER = b.Q(i, j, k, 4);
        double &EL = b.Q(i, j, k - 1, 4);

        // wave speed estimate
        double lam =
            fmax(abs(UL) + b.qh(i, j, k, 3), abs(UR) + b.qh(i, j, k - 1, 3)) *
            b.kS(i, j, k);
        UR *= b.kS(i, j, k);
        UL *= b.kS(i, j, k);

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
        double FYiR, FYiL;
        double YiR, YiL;
        for (int n = 0; n < b.ne - 5; n++) {
          FYiR = b.Q(i, j, k, 5 + n) * UR;
          FYiL = b.Q(i, j, k - 1, 5 + n) * UL;
          YiR = b.Q(i, j, k, 5 + n);
          YiL = b.Q(i, j, k - 1, 5 + n);
          b.kF(i, j, k, 5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
        }
      });
}

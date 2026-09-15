#include "faces.hpp"

PG_RANGE(faces)
struct rusanov {
  inR QR, qR, qhR;
  inL QL, qL, qhL;
  out F;
  in A;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    double UR;
    double UL;

    const double &ufR = qR(1);
    const double &vfR = qR(2);
    const double &wfR = qR(3);

    const double &ufL = qL(1);
    const double &vfL = qL(2);
    const double &wfL = qL(3);

    UR = nx * ufR + ny * vfR + nz * wfR;
    UL = nx * ufL + ny * vfL + nz * wfL;

    const double &rhoR = QR(0);
    const double &rhoL = QL(0);

    const double &pR = qR(0);
    const double &pL = qL(0);

    const double &ER = QR(4);
    const double &EL = QL(4);

    // wave speed estimate
    double lam = fmax(abs(UL) + qhR(3), abs(UR) + qhL(3)) * S;
    UR *= S;
    UL *= S;

    // Continuity rho*Ui
    double FrhoR, FrhoL;
    FrhoR = UR * rhoR;
    FrhoL = UL * rhoL;
    F(0) = 0.5 * (FrhoR + FrhoL - lam * (rhoR - rhoL));

    double FUR, FUL;
    // x momentum rho*u*Ui+ p*Ax
    FUR = UR * ufR * rhoR + pR * A(0);
    FUL = UL * ufL * rhoL + pL * A(0);
    F(1) = 0.5 * (FUR + FUL - lam * (rhoR * ufR - rhoL * ufL));

    // y momentum rho*v*Ui+ p*Ay
    FUR = UR * vfR * rhoR + pR * A(1);
    FUL = UL * vfL * rhoL + pL * A(1);
    F(2) = 0.5 * (FUR + FUL - lam * (rhoR * vfR - rhoL * vfL));

    // w momentum rho*w*Ui+ p*Az
    FUR = UR * wfR * rhoR + pR * A(2);
    FUL = UL * wfL * rhoL + pL * A(2);
    F(3) = 0.5 * (FUR + FUL - lam * (rhoR * wfR - rhoL * wfL));

    // Total energy (rhoE+ p)*Ui)
    double FER, FEL;
    FER = UR * (ER + pR);
    FEL = UL * (EL + pL);
    F(4) = 0.5 * (FER + FEL - lam * (ER - EL));

    // Species
    double FYiR, FYiL;
    double YiR, YiL;
    for (int n = 0; n < ne - 5; n++) {
      FYiR = QR(5 + n) * UR;
      FYiL = QL(5 + n) * UL;
      YiR = QR(5 + n);
      YiL = QL(5 + n);
      F(5 + n) = 0.5 * (FYiR + FYiL - lam * (YiR - YiL));
    }
  }
};

PG_ABI void pgRusanov(const rusanov &k, const pgTiling &t) {
  forCells("rusanov face conv fluxes", t, k);
}

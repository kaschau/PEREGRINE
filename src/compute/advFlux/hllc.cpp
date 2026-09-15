#include "faces.hpp"

PG_RANGE(faces)
struct hllc {
  inL QL, qL, qhL;
  inR QR, qR, qhR;
  out F;
  in A;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double S, nx, ny, nz;
    faceNormal(A(0), A(1), A(2), S, nx, ny, nz);

    const double &ufR = qR(1);
    const double &vfR = qR(2);
    const double &wfR = qR(3);

    const double &ufL = qL(1);
    const double &vfL = qL(2);
    const double &wfL = qL(3);

    double UR = nx * ufR + ny * vfR + nz * wfR;
    double UL = nx * ufL + ny * vfL + nz * wfL;

    const double &rhoR = QR(0);
    const double &rhoL = QL(0);

    const double &rhouR = QR(1);
    const double &rhouL = QL(1);
    const double &rhovR = QR(2);
    const double &rhovL = QL(2);
    const double &rhowR = QR(3);
    const double &rhowL = QL(3);

    const double &pR = qR(0);
    const double &pL = qL(0);

    const double &ER = QR(4);
    const double &EL = QL(4);

    const double &cR = qhR(3);
    const double &cL = qhL(3);

    double pstar = 0.5 * (pL + pR) -
                   0.5 * (UR - UL) * 0.5 * (rhoL + rhoR) * 0.5 * (cL + cR);
    pstar = fmax(0.0, pstar);

    // wave speed estimate
    double SL = UL - cL;
    double SR = UR + cR;
    double Sstar = (pR - pL + rhoL * UL * (SL - UL) - rhoR * UR * (SR - UR)) /
                   (rhoL * (SL - UL) - rhoR * (SR - UR));

    if (SL >= 0.0) {
      F(0) = UL * rhoL * S;
      F(1) = UL * rhouL * S + pL * A(0);
      F(2) = UL * rhovL * S + pL * A(1);
      F(3) = UL * rhowL * S + pL * A(2);
      F(4) = UL * (EL + pL) * S;
      for (int n = 0; n < ne - 5; n++) {
        double rhoYiL = QL(5 + n);
        F(5 + n) = UL * rhoYiL * S;
      }
    } else if ((SL <= 0.0) && (Sstar >= 0.0)) {
      double FrhoL, FUL, FVL, FWL, FEL, UstarL;
      FrhoL = UL * rhoL * S;
      FUL = UL * rhouL * S + pL * A(0);
      FVL = UL * rhovL * S + pL * A(1);
      FWL = UL * rhowL * S + pL * A(2);
      FEL = UL * (EL + pL) * S;
      UstarL = rhoL * (SL - UL) / (SL - Sstar);

      F(0) = FrhoL + SL * (UstarL - rhoL) * S;
      F(1) = FUL + SL * (UstarL * Sstar * nx - rhouL) * S;
      F(2) = FVL + SL * (UstarL * Sstar * ny - rhovL) * S;
      F(3) = FWL + SL * (UstarL * Sstar * nz - rhowL) * S;
      F(4) = FEL +
             SL *
                 (UstarL * (EL / rhoL +
                            (Sstar - UL) * (Sstar + pL / (rhoL * (SL - UL)))) -
                  EL) *
                 S;
      for (int n = 0; n < ne - 5; n++) {
        double FYiL, YiL, rhoYiL;
        FYiL = QL(5 + n) * UL * S;
        YiL = qL(5 + n);
        rhoYiL = QL(5 + n);
        F(5 + n) = FYiL + SL * (UstarL * YiL - rhoYiL) * S;
      }
    } else if ((SR >= 0.0) && (Sstar <= 0.0)) {
      double FrhoR, FUR, FVR, FWR, FER, UstarR;
      FrhoR = UR * rhoR * S;
      FUR = UR * rhouR * S + pR * A(0);
      FVR = UR * rhovR * S + pR * A(1);
      FWR = UR * rhowR * S + pR * A(2);
      FER = UR * (ER + pR) * S;
      UstarR = rhoR * (SR - UR) / (SR - Sstar);

      F(0) = FrhoR + SR * (UstarR - rhoR) * S;
      F(1) = FUR + SR * (UstarR * Sstar * nx - rhouR) * S;
      F(2) = FVR + SR * (UstarR * Sstar * ny - rhovR) * S;
      F(3) = FWR + SR * (UstarR * Sstar * nz - rhowR) * S;
      F(4) = FER +
             SR *
                 (UstarR * (ER / rhoR +
                            (Sstar - UR) * (Sstar + pR / (rhoR * (SR - UR)))) -
                  ER) *
                 S;
      for (int n = 0; n < ne - 5; n++) {
        double FYiR, YiR, rhoYiR;
        FYiR = QR(5 + n) * UR * S;
        YiR = qR(5 + n);
        rhoYiR = QR(5 + n);
        F(5 + n) = FYiR + SR * (UstarR * YiR - rhoYiR) * S;
      }
    } else if (SR <= 0.0) {
      F(0) = UR * rhoR * S;
      F(1) = UR * rhouR * S + pR * A(0);
      F(2) = UR * rhovR * S + pR * A(1);
      F(3) = UR * rhowR * S + pR * A(2);
      F(4) = UR * (ER + pR) * S;
      for (int n = 0; n < ne - 5; n++) {
        double rhoYiR = QR(5 + n);
        F(5 + n) = UR * rhoYiR * S;
      }
    }
  }
};

PG_ABI void pgHllc(const hllc &k, const pgTiling &t) {
  forCells("hllc face conv fluxes", t, k);
}

#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "thermo/eos.hpp"
#include "vector"

PG_RANGE(cellCenters)
struct invertDQ {
  cellVecIn Q, dIJK, q, qh, qt;
  cellScalIn dtau;
  cellVecInOut dQ;
  caseIn dt;
  bool viscous;
  KOKKOS_INLINE_FUNCTION void operator()() const {
#if !defined(PG_LOW_MACH) && !defined(PG_CHEMISTRY_JACOBIAN)
    // unpreconditioned and no source Jacobian: Gamma is dQ/dq, so the pseudo
    // system is the scalar (1 / dtau + 3 / (2 dt)) dQ = R in conserved
    // variables, and the increment scaled by the pseudo step as below
    const fpdtype scale = dtau / (1.0 + 3.0 / 2.0 * dtau / dt());
    for (int l = 0; l < ne; l++) {
      dQ(l) *= scale;
    }
#else
    //-------------------------------------------------------------------------------------------|
    // Solve (Gamma + 3 dtau / (2 dt) dQ/dq - dtau dS/dq) dq = dQ for dq
    //
    // The premultiplying matrix takes the form of Weiss and Smith
    //
    // Preconditioning applied to variable and constant density flows
    // Weiss, Jonathan M. and Smith, Wayne A.
    // AIAA Journal
    // 1995
    // doi: 10.2514/3.12946
    //
    // The matrix is never formed: it is S + Qhat g^T, Qhat the conserved
    // state over the density, (1, u, v, w, H, Y), g the density's gradient
    // over the primitives with Gamma's Theta in the pressure slot, and S
    // sparse -- so Sherman and Morrison solve it from two solves of S,
    // each linear in the species count, with no pivoting.
    //-------------------------------------------------------------------------------------------|
    const fpdtype &p = q(0);
    const fpdtype &T = q(1);
    const fpdtype &rho = Q(0);
    const fpdtype rhoinv = 1.0 / rho;
    const fpdtype u = Q(1) * rhoinv;
    const fpdtype v = Q(2) * rhoinv;
    const fpdtype w = Q(3) * rhoinv;
    fpdtype Y[ns];
    fpdtype rho_Y[ns];
    fpdtype cp = qh(1);
    fpdtype H = qh(2) * rhoinv + 0.5 * (u * u + v * v + w * w);
    fpdtype c = qh(3);
    massFractions(Q, rhoinv, Y);

    // the density's derivatives, and the species enthalpies, from the eos
    fpdtype rho_p, rho_T;
    eos::densityDerivatives(
        p, T, rho, [&](const int n) { return Y[n]; }, rho_p, rho_T, rho_Y);
    const auto hi = eos::enthalpies(T, qh);

    // Gamma's multiplier is one, the transformation's 3 dtau / (2 dt)
    const fpdtype m = 3.0 / 2.0 * dtau / dt();
#ifdef PG_LOW_MACH
    // Reference velocity for preconditioning theta
    const fpdtype U = sqrt(u * u + v * v + w * w);
    const fpdtype nu = viscous ? qt(0) / Q(0) : 0.0;
    const fpdtype &dI = dIJK(0);
    const fpdtype &dJ = dIJK(1);
    const fpdtype &dK = dIJK(2);
    const fpdtype Ur = referenceVelocity(U, c, nu, dI, dJ, dK);

    const fpdtype Theta = 1.0 / pow(Ur, 2.0) - rho_T / (rho * cp);
#else
    // unpreconditioned: Gamma is dQ/dq
    const fpdtype Theta = rho_p;
#endif
    // S takes the whole pressure column, Qhat times ThetaBar, so it is
    // invertible; g then runs over the temperature and the species
    const fpdtype ThetaBar = Theta + m * rho_p, M = 1.0 + m, Mrho = M * rho;
    // S's species rows: their own entry, their temperature entry and the
    // energy row's enthalpy differences, the chemistry's Jacobian taken off
    // the first two where the case has one
    fpdtype d[ns], Tcol[ns], hy[ns];
    for (int n = 0; n < ns - 1; n++) {
      d[n] = Mrho;
      Tcol[n] = 0.0;
      hy[n] = Mrho * (hi(n) - hi(ns - 1));
    }
#ifdef PG_CHEMISTRY_JACOBIAN
    // less dtau times the entries the rung evaluates, each species row's
    // own and its temperature's, at fixed density
    chemistry::jacobianOf<chemistry::PG_CHEMISTRY_JACOBIAN>(
        Q, q, [&](const int j, const int col, const fpdtype v) {
          (col == 4 ? Tcol[j] : d[j]) -= dtau * v;
        });
#endif
    // S y = r for the two right-hand sides, dQ and Qhat: the pressure row
    // alone, the momenta, then the temperature row with the species rows
    // folded into it, then the species
    const fpdtype S40 = ThetaBar * H + M * T * rho_T * rhoinv;
    fpdtype rb[ne], rQ[ne], yb[ne], yQ[ne];
    for (int l = 0; l < ne; l++)
      rb[l] = dQ(l);
    rQ[0] = 1.0, rQ[1] = u, rQ[2] = v, rQ[3] = w, rQ[4] = H;
    for (int n = 0; n < ns - 1; n++)
      rQ[5 + n] = Y[n];
    const fpdtype *rs[2] = {rb, rQ};
    fpdtype *ys[2] = {yb, yQ};
    for (int k = 0; k < 2; k++) {
      const fpdtype *r = rs[k];
      fpdtype *y = ys[k];
      y[0] = r[0] / ThetaBar;
      y[1] = (r[1] - ThetaBar * u * y[0]) / Mrho;
      y[2] = (r[2] - ThetaBar * v * y[0]) / Mrho;
      y[3] = (r[3] - ThetaBar * w * y[0]) / Mrho;
      fpdtype num = r[4] - S40 * y[0] - Mrho * (u * y[1] + v * y[2] + w * y[3]);
      fpdtype den = Mrho * cp;
      for (int n = 0; n < ns - 1; n++) {
        y[5 + n] = (r[5 + n] - ThetaBar * Y[n] * y[0]) / d[n];
        num -= hy[n] * y[5 + n];
        den -= hy[n] * Tcol[n] / d[n];
      }
      y[4] = num / den;
      for (int n = 0; n < ns - 1; n++)
        y[5 + n] -= Tcol[n] * y[4] / d[n];
    }
    // Sherman and Morrison: x = yb - yQ (g . yb) / (1 + g . yQ)
    fpdtype gb = rho_T * yb[4], gQ = rho_T * yQ[4];
    for (int n = 0; n < ns - 1; n++) {
      gb += rho_Y[n] * yb[5 + n];
      gQ += rho_Y[n] * yQ[5 + n];
    }
    const fpdtype ratio = M * gb / (1.0 + M * gQ);
    // scaled by the cell's pseudo step, so a stage adds it as it is
    for (int l = 0; l < ne; l++)
      dQ(l) = (yb[l] - ratio * yQ[l]) * dtau;

    fpdtype tempRow[ne];
    // the increment back in conserved variables, dQ = (dQ/dq) dq, the
    // transformation's columns applied one at a time
    for (int l = 0; l < ne; l++) {
      tempRow[l] = 0.0;
    }
    {
      const fpdtype dp = dQ(0), du = dQ(1), dv = dQ(2), dw = dQ(3), dT = dQ(4);
      fpdtype drho = rho_p * dp + rho_T * dT;
      fpdtype dE = (rho_p * H + T * rho_T / rho) * dp +
                   rho * (u * du + v * dv + w * dw) +
                   (rho_T * H + rho * cp) * dT;
      for (int n = 5; n < ne; n++) {
        const fpdtype dY = dQ(n);
        drho += rho_Y[n - 5] * dY;
        dE += (H * rho_Y[n - 5] + rho * (hi(n - 5) - hi(ns - 1))) * dY;
      }
      tempRow[0] = drho;
      tempRow[1] = drho * u + rho * du;
      tempRow[2] = drho * v + rho * dv;
      tempRow[3] = drho * w + rho * dw;
      tempRow[4] = dE;
      for (int n = 5; n < ne; n++) {
        tempRow[n] = drho * Y[n - 5] + rho * dQ(n);
      }
    }
    for (int l = 0; l < ne; l++) {
      dQ(l) = tempRow[l];
    }
#endif
  }
};

PG_ABI void pgInvertDQ(const invertDQ &k, const pgTiling &t) {
  forCells("dQ = dQdq (Gamma + dqdQ)^{-1} dQ", t, k);
}

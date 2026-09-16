#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "thermo/eos.hpp"
#include "vector"

PG_RANGE(cellCenters)
struct invertDQ {
  cellCenterIn Q, dIJK, dtau, q, qh, qt;
  cellCenterInOut dQ;
  dims d;
  double dt;
  bool viscous;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
    //-------------------------------------------------------------------------------------------|
    // Solve (\Gamma + dqdQ) dq = dQ to solver for dqdt
    //
    // The premultiplying matrix takes the form of Weiss and Smith
    //
    // Preconditioning applied to variable and constant density flows
    // Weiss, Jonathan M. and Smith, Wayne A.
    // AIAA Journal
    // 1995
    // doi: 10.2514/3.12946
    //-------------------------------------------------------------------------------------------|
    double iMult = 1.0;
    double jMult = 1.0;
    double kMult = 1.0;
    if (ni == 2) {
      iMult = Kokkos::Experimental::infinity<double>::value;
    }
    if (nj == 2) {
      jMult = Kokkos::Experimental::infinity<double>::value;
    }
    if (nk == 2) {
      kMult = Kokkos::Experimental::infinity<double>::value;
    }

    double GdQ[ne][ne];
    int perm[ne];
    double tempRow[ne];

    ////////////////////////////////////////////////
    ///// COMPUTE GdQ MATRIX
    ///// Sum of Preconditioning matrix and
    ///// convservative to primative variable
    ///// transformation
    /////
    /////  \Gamma + 3d[e]tau / (2d[e]t) dQdq
    /////
    ////////////////////////////////////////////////
    const double &p = q(0);
    const double &T = q(1);
    const double &rho = Q(0);
    const double rhoinv = 1.0 / rho;
    const double u = Q(1) * rhoinv;
    const double v = Q(2) * rhoinv;
    const double w = Q(3) * rhoinv;
    double Y[ns];
    double rho_Y[ns];
    double cp = qh(1);
    double H = qh(2) * rhoinv + 0.5 * (u * u + v * v + w * w);
    double c = qh(3);
    massFractions(Q, rhoinv, Y);

    // the density's derivatives, and the species enthalpies, from the eos
    double rho_p, rho_T;
    eos::densityDerivatives(
        p, T, rho, [&](const int n) { return Y[n]; }, rho_p, rho_T, rho_Y);
    const auto hi = eos::enthalpies(T, qh);

    /////////////////////////////////////////////////
    // The preconditioning and transformation matrix
    // share a very similar form, only differing by
    // the multiplier of the first column, and the
    // multiplication of the time derivatives for
    // the prim/cons transformation matrix.
    /////////////////////////////////////////////////
    for (int l = 0; l < ne; l++) {
      for (int m = 0; m < ne; m++) {
        GdQ[l][m] = 0.0;
      }
    }
    double Thetas[2];
    double mults[2];

    // Prematrix multipliers (constants)
    mults[0] = 1.0;
    mults[1] = 3.0 / 2.0 * dtau() / dt;

    // Reference velocity for preconditioning theta
    const double U = sqrt(u * u + v * v + w * w);
    const double nu = viscous ? qt(0) / Q(0) : 0.0;
    const double &dI = dIJK(0);
    const double &dJ = dIJK(1);
    const double &dK = dIJK(2);
    const double Ur =
        referenceVelocity(U, c, nu, iMult * dI, jMult * dJ, kMult * dK);

    // Thetas (just rho_p for dQdq)
    Thetas[0] = 1.0 / pow(Ur, 2.0) - rho_T / (rho * cp);
    Thetas[1] = rho_p;

    ///////////////////////////////////////////////////////////////////
    // Gamma and dQdq are constricted in the following blocks
    // |-----------------------|------------------|
    // |                       |                  |
    // |         (1)           |       (2)        |
    // |      Single Comp      |  Prims/Species   |
    // |       Primatives      |                  |
    // |                       |                  |
    // |-----------------------|------------------|
    // |                       |                  |
    // |         (3)           |       (4)        |
    // |     Species/Prims     | Species/Species  |
    // |                       |                  |
    // |                       |                  |
    // |-----------------------|------------------|
    //
    // In a column by column manner
    ///////////////////////////////////////////////////////////////////

    for (int p = 0; p < 2; p++) {
      double Theta = Thetas[p];
      double mult = mults[p];

      // Block (1)
      // First column
      GdQ[0][0] += mult * Theta;
      GdQ[1][0] += mult * Theta * u;
      GdQ[2][0] += mult * Theta * v;
      GdQ[3][0] += mult * Theta * w;
      GdQ[4][0] += mult * (Theta * H + T * rho_T / rho);

      // Second column
      GdQ[0][1] += mult * 0.0;
      GdQ[1][1] += mult * rho;
      GdQ[2][1] += mult * 0.0;
      GdQ[3][1] += mult * 0.0;
      GdQ[4][1] += mult * rho * u;

      // Third column
      GdQ[0][2] += mult * 0.0;
      GdQ[1][2] += mult * 0.0;
      GdQ[2][2] += mult * rho;
      GdQ[3][2] += mult * 0.0;
      GdQ[4][2] += mult * rho * v;

      // Fourth column
      GdQ[0][3] += mult * 0.0;
      GdQ[1][3] += mult * 0.0;
      GdQ[2][3] += mult * 0.0;
      GdQ[3][3] += mult * rho;
      GdQ[4][3] += mult * rho * w;

      // Fifth column
      GdQ[0][4] += mult * rho_T;
      GdQ[1][4] += mult * rho_T * u;
      GdQ[2][4] += mult * rho_T * v;
      GdQ[3][4] += mult * rho_T * w;
      GdQ[4][4] += mult * (rho_T * H + rho * cp);

      for (int n = 5; n < ne; n++) {
        // Block (2) nth column
        GdQ[0][n] += mult * rho_Y[n - 5];
        GdQ[1][n] += mult * rho_Y[n - 5] * u;
        GdQ[2][n] += mult * rho_Y[n - 5] * v;
        GdQ[3][n] += mult * rho_Y[n - 5] * w;
        double h_y = hi(n - 5) - hi(ns - 1);
        GdQ[4][n] += mult * (H * rho_Y[n - 5] + rho * h_y);
        // Block (3)
        GdQ[n][0] += mult * Theta * Y[n - 5];
        GdQ[n][1] += mult * 0.0;
        GdQ[n][2] += mult * 0.0;
        GdQ[n][3] += mult * 0.0;
        GdQ[n][4] += mult * rho_T * Y[n - 5];
      }

      // Block (4)
      for (int n = 5; n < ne; n++) {
        for (int q = 5; q < ne; q++) {
          GdQ[q][n] += mult * Y[q - 5] * rho_Y[n - 5];
        }
      }
      for (int n = 5; n < ne; n++) {
        GdQ[n][n] += mult * rho;
      }
    }

    /////////////////////////////////////////////////////////////////////////////
    // Perform LU decomposition with partial pivoting
    // Routine modifies GdQ in place resulting in a
    // strictly lower triangle matrix with 1.0 along the diagonal
    // and an upper triangular matrix including the diagonal.
    /////////////////////////////////////////////////////////////////////////////

    for (int l = 0; l < ne; l++) {
      perm[l] = l;
    }

    for (int l = 0; l < ne; l++) {
      int pivotInd = 0;
      double pivot = 0.0;
      int tempInd;
      for (int m = l; m < ne; m++)
        if (abs(GdQ[m][l]) > abs(pivot)) {
          pivot = GdQ[m][l];
          pivotInd = m;
        }

      for (int p = 0; p < ne; p++) {
        tempRow[p] = GdQ[l][p];
        GdQ[l][p] = GdQ[pivotInd][p];
        GdQ[pivotInd][p] = tempRow[p];
      }

      tempInd = perm[l];
      perm[l] = perm[pivotInd];
      perm[pivotInd] = tempInd;

      for (int p = l + 1; p < ne; p++) {
        double temp;
        temp = GdQ[p][l] /= GdQ[l][l];
        for (int q = l + 1; q < ne; q++) {
          GdQ[p][q] -= temp * GdQ[l][q];
        }
      }
    }

    // Row permute dQ to match LU
    for (int l = 0; l < ne; l++) {
      tempRow[l] = dQ(perm[l]);
    }
    for (int l = 0; l < ne; l++) {
      dQ(l) = tempRow[l];
    }

    // Solve Ax = b where A = LU by first solving for
    //
    // Lz = a then Ux=z
    //
    // Form of the equations is actually
    //
    // LU(dq) = dQ
    //
    // So begin with Lz = dQ where tempRow = z

    for (int l = 0; l < ne; l++) {
      for (int q = 0; q < l; q++) {
        tempRow[l] -= GdQ[l][q] * tempRow[q];
      }
    }

    // Now solve Ux=z which is actually
    //
    // U(dq) = tempRow
    //
    // Recall we are working with primatives so we will modify the dQ
    // view in place with the resultant dq values (as x)

    for (int l = ne - 1; l > -1; l--) {
      dQ(l) = tempRow[l];
      for (int q = ne - 1; q > l; q--) {
        dQ(l) -= GdQ[l][q] * dQ(q);
      }
      dQ(l) /= GdQ[l][l];
    }
    // scaled by the cell's pseudo step, so a stage adds it as it is
    for (int l = 0; l < ne; l++) {
      dQ(l) *= dtau();
    }

    // the increment back in conserved variables, dQ = (dQ/dq) dq, the
    // transformation's columns applied one at a time
    for (int l = 0; l < ne; l++) {
      tempRow[l] = 0.0;
    }
    {
      const double dp = dQ(0), du = dQ(1), dv = dQ(2), dw = dQ(3), dT = dQ(4);
      double drho = rho_p * dp + rho_T * dT;
      double dE = (rho_p * H + T * rho_T / rho) * dp +
                  rho * (u * du + v * dv + w * dw) +
                  (rho_T * H + rho * cp) * dT;
      for (int n = 5; n < ne; n++) {
        const double dY = dQ(n);
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
  }
};

PG_ABI void pgInvertDQ(const invertDQ &k, const pgTiling &t) {
  forCells("dQ = dQdq (Gamma + dqdQ)^{-1} dQ", t, k);
}

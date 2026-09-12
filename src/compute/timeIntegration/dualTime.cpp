#include "array"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------------------------|
//
// Dual Time with preconditioning from
//
//     Preconditioning applied to variable and constant density flows
//     Weiss, Jonathan M. and Smith, Wayne A.
//     AIAA Journal
//     1995
//     doi: 10.2514/3.12946
//
//---------------------------------------------------------------------------------------------|

// Weiss & Smith reference velocity: the flow speed, clipped between eps*c and
// c, and no smaller than the diffusion velocity nu/dx. An inviscid call passes
// nu = 0 and a degenerate direction an infinite length, so neither binds.
static KOKKOS_INLINE_FUNCTION double
referenceVelocity(const double U, const double c, const double nu,
                  const double dI, const double dJ, const double dK) {
  const double eps = 1.0e-5;
  double Ur = fmax(U, eps * c);
  Ur = fmax(Ur, nu / dI);
  Ur = fmax(Ur, nu / dJ);
  Ur = fmax(Ur, nu / dK);
  return fmin(Ur, c);
}

PG_ABI void pgDQdt(const pgView *Q_, const pgView *Qn_, const pgView *Qnm1_,
                   const pgView *dQ_, const pgDims *d, double dt) {
  auto Q = as4(*Q_);
  auto Qn = as4(*Qn_);
  auto Qnm1 = as4(*Qnm1_);
  auto dQ = as4(*dQ_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Add to dQ with real time derivative source term
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "dQdt", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        dQ(i, j, k, l) -=
            (3.0 * Q(i, j, k, l) - 4.0 * Qn(i, j, k, l) + Qnm1(i, j, k, l)) /
            (2 * dt);
      });
}

PG_ABI void pgLocalDtau(const pgView *Q_, const pgView *dIJK_,
                        const pgView *dtau_, const pgView *iS_,
                        const pgView *jS_, const pgView *kS_, const pgView *q_,
                        const pgView *qh_, const pgView *qt_, const pgDims *d,
                        bool viscous) {
  auto Q = as4(*Q_);
  auto dIJK = as4(*dIJK_);
  auto dtau = as3(*dtau_);
  auto iS = as4(*iS_);
  auto jS = as4(*jS_);
  auto kS = as4(*kS_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  auto qt = as4(*qt_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Compute local pseudo time step
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

  MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});
  Kokkos::parallel_for(
      "localDtau", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Cell lengths
        const double &dI = dIJK(i, j, k, 0);
        const double &dJ = dIJK(i, j, k, 1);
        const double &dK = dIJK(i, j, k, 2);

        // Find max convective CFL
        double S0, S1;
        double inx0, iny0, inz0, inx1, iny1, inz1;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S0, inx0,
                   iny0, inz0);
        faceNormal(iS(i + 1, j, k, 0), iS(i + 1, j, k, 1), iS(i + 1, j, k, 2),
                   S1, inx1, iny1, inz1);
        double jnx0, jny0, jnz0, jnx1, jny1, jnz1;
        faceNormal(jS(i, j, k, 0), jS(i, j, k, 1), jS(i, j, k, 2), S0, jnx0,
                   jny0, jnz0);
        faceNormal(jS(i, j + 1, k, 0), jS(i, j + 1, k, 1), jS(i, j + 1, k, 2),
                   S1, jnx1, jny1, jnz1);
        double knx0, kny0, knz0, knx1, kny1, knz1;
        faceNormal(kS(i, j, k, 0), kS(i, j, k, 1), kS(i, j, k, 2), S0, knx0,
                   kny0, knz0);
        faceNormal(kS(i, j, k + 1, 0), kS(i, j, k + 1, 1), kS(i, j, k + 1, 2),
                   S1, knx1, kny1, knz1);
        const double &u = q(i, j, k, 1);
        const double &v = q(i, j, k, 2);
        const double &w = q(i, j, k, 3);

        double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                         pow(0.5 * (iny0 + iny1) * v, 2.0) +
                         pow(0.5 * (inz0 + inz1) * w, 2.0));
        double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                         pow(0.5 * (jny0 + jny1) * v, 2.0) +
                         pow(0.5 * (jnz0 + jnz1) * w, 2.0));
        double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                         pow(0.5 * (kny0 + kny1) * v, 2.0) +
                         pow(0.5 * (knz0 + knz1) * w, 2.0));

        const double &c = qh(i, j, k, 3);

        double pseudoCFL = 0.5;
        double pseudoVNN = 0.1;

        // the preconditioned system's wave speeds set the pseudo step
        const double nu = viscous ? qt(i, j, k, 0) / Q(i, j, k, 0) : 0.0;
        const double Ur = referenceVelocity(sqrt(u * u + v * v + w * w), c, nu,
                                            iMult * dI, jMult * dJ, kMult * dK);
        // the preconditioned system propagates u' +- c', not u + c
        const double alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
        const double a2 = alpha * alpha;
        const double Ur2 = Ur * Ur;

        double dtauCell = Kokkos::Experimental::infinity<double>::value;
        dtauCell = fmin(
            dtauCell, iMult * pseudoCFL * dI /
                          (abs((1.0 - alpha) * uI) + sqrt(a2 * uI * uI + Ur2)));
        dtauCell = fmin(
            dtauCell, jMult * pseudoCFL * dJ /
                          (abs((1.0 - alpha) * uJ) + sqrt(a2 * uJ * uJ + Ur2)));
        dtauCell = fmin(
            dtauCell, kMult * pseudoCFL * dK /
                          (abs((1.0 - alpha) * uK) + sqrt(a2 * uK * uK + Ur2)));
        if (viscous) {
          dtauCell = fmin(dtauCell, iMult * pseudoVNN * pow(dI, 2.0) / nu);
          dtauCell = fmin(dtauCell, jMult * pseudoVNN * pow(dJ, 2.0) / nu);
          dtauCell = fmin(dtauCell, kMult * pseudoVNN * pow(dK, 2.0) / nu);
        }

        dtau(i, j, k) = dtauCell;
      });
}

PG_ABI void pgDTrk3s1(const pgView *Q0_, const pgView *dQ_, const pgView *dtau_,
                      const pgView *q_, const pgDims *d) {
  auto Q0 = as4(*Q0_);
  auto dQ = as4(*dQ_);
  auto dtau = as3(*dtau_);
  auto q = as4(*q_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "DTrk3 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // store zeroth stage
        Q0(i, j, k, l) = q(i, j, k, l);
        q(i, j, k, l) += dtau(i, j, k) * dQ(i, j, k, l);
      });
}

PG_ABI void pgDTrk3s2(const pgView *Q0_, const pgView *dQ_, const pgView *dtau_,
                      const pgView *q_, const pgDims *d) {
  auto Q0 = as4(*Q0_);
  auto dQ = as4(*dQ_);
  auto dtau = as3(*dtau_);
  auto q = as4(*q_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "DTrk3 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        q(i, j, k, l) = 0.75 * Q0(i, j, k, l) + 0.25 * q(i, j, k, l) +
                        0.25 * dQ(i, j, k, l) * dtau(i, j, k);
      });
}

PG_ABI void pgDTrk3s3(const pgView *Q0_, const pgView *dQ_, const pgView *dtau_,
                      const pgView *q_, const pgDims *d) {
  auto Q0 = as4(*Q0_);
  auto dQ = as4(*dQ_);
  auto dtau = as3(*dtau_);
  auto q = as4(*q_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 3
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "DTrk3 stage 3", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        q(i, j, k, l) = (Q0(i, j, k, l) + 2.0 * q(i, j, k, l) +
                         2.0 * dQ(i, j, k, l) * dtau(i, j, k)) /
                        3.0;
      });
}

PG_ABI void pgInvertDQ(const pgView *Q_, const pgView *dIJK_, const pgView *dQ_,
                       const pgView *dtau_, const pgView *q_, const pgView *qh_,
                       const pgView *qt_, const pgView *MW_, double Ru,
                       const pgDims *d, double dt, bool viscous) {
  auto Q = as4(*Q_);
  auto dIJK = as4(*dIJK_);
  auto dQ = as4(*dQ_);
  auto dtau = as3(*dtau_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  auto qt = as4(*qt_);
  auto MW = as1(*MW_);
  const int ns = MW.extent(0);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);
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

  MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  threeDview GdQ("GdQ", numIds, ne, ne);
  twoDview tempRow("tempRow", numIds, ne);
  twoDviewInt perm("perm", numIds, ne);
#endif

#ifdef NSCOMPILE
#define GdQ(INDEX, INDEX1) GdQ[INDEX][INDEX1]
#define perm(INDEX) perm[INDEX]
#define tempRow(INDEX) tempRow[INDEX]
#define ne 5 + NS - 1
#else
#define GdQ(INDEX, INDEX1) GdQ(id, INDEX, INDEX1)
#define perm(INDEX) perm(id, INDEX)
#define tempRow(INDEX) tempRow(id, INDEX)
#endif

#ifndef NSCOMPILE
  twoDview Y("Y", numIds, ns);
  twoDview rho_Y("rho_Y", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define rho_Y(INDEX) rho_Y[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define rho_Y(INDEX) rho_Y(id, INDEX)
#endif

  Kokkos::parallel_for(
      "dq = (Gamma + dqdQ)^{-1} dQ", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

#ifdef NSCOMPILE
        double GdQ(ne, ne);
        int perm(ne);
        double tempRow(ne);
#else
        int id = token.acquire();
#endif

        ////////////////////////////////////////////////
        ///// COMPUTE GdQ MATRIX
        ///// Sum of Preconditioning matrix and
        ///// convservative to primative variable
        ///// transformation
        /////
        /////  \Gamma + 3*dtau / (2*dt) dQdq
        /////
        ////////////////////////////////////////////////
        const double &p = q(i, j, k, 0);
        const double &u = q(i, j, k, 1);
        const double &v = q(i, j, k, 2);
        const double &w = q(i, j, k, 3);
        const double &T = q(i, j, k, 4);
        const double &rho = Q(i, j, k, 0);
#ifdef NSCOMPILE
        double Y(ns);
        double rho_Y(ns);
#endif
        double cp = qh(i, j, k, 1);
        double H = qh(i, j, k, 2) / rho +
                   0.5 * (pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0));
        double c = qh(i, j, k, 3);
        // Compute nth species Y
        Y(ns - 1) = 1.0;
        double denom = 0.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
          denom += Y(n) / MW(n);
        }
        denom += Y(ns - 1) / MW(ns - 1);

        // Compute MWmix
        double MWmix = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double X = Y(n) / MW(n) / denom;
          MWmix += MW(n) * X;
        }

        // Compute required derivatives
        double rho_p = rho / p;
        double rho_T = -rho / T;

        for (int n = 0; n < ns - 1; n++) {
          rho_Y(n) = -rho * (MWmix * (1.0 / MW(n) - 1.0 / MW(ns - 1)));
        }

        /////////////////////////////////////////////////
        // The preconditioning and transformation matrix
        // share a very similar form, only differing by
        // the multiplier of the first column, and the
        // multiplication of the time derivatives for
        // the prim/cons transformation matrix.
        /////////////////////////////////////////////////
        for (int l = 0; l < ne; l++) {
          for (int m = 0; m < ne; m++) {
            GdQ(l, m) = 0.0;
          }
        }
        double Thetas[2];
        double mults[2];

        // Prematrix multipliers (constants)
        mults[0] = 1.0;
        mults[1] = 3.0 / 2.0 * dtau(i, j, k) / dt;

        // Reference velocity for preconditioning theta
        const double U = sqrt(u * u + v * v + w * w);
        const double nu = viscous ? qt(i, j, k, 0) / Q(i, j, k, 0) : 0.0;
        const double &dI = dIJK(i, j, k, 0);
        const double &dJ = dIJK(i, j, k, 1);
        const double &dK = dIJK(i, j, k, 2);
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
          GdQ(0, 0) += mult * Theta;
          GdQ(1, 0) += mult * Theta * u;
          GdQ(2, 0) += mult * Theta * v;
          GdQ(3, 0) += mult * Theta * w;
          GdQ(4, 0) += mult * (Theta * H + T * rho_T / rho);

          // Second column
          GdQ(0, 1) += mult * 0.0;
          GdQ(1, 1) += mult * rho;
          GdQ(2, 1) += mult * 0.0;
          GdQ(3, 1) += mult * 0.0;
          GdQ(4, 1) += mult * rho * u;

          // Third column
          GdQ(0, 2) += mult * 0.0;
          GdQ(1, 2) += mult * 0.0;
          GdQ(2, 2) += mult * rho;
          GdQ(3, 2) += mult * 0.0;
          GdQ(4, 2) += mult * rho * v;

          // Fourth column
          GdQ(0, 3) += mult * 0.0;
          GdQ(1, 3) += mult * 0.0;
          GdQ(2, 3) += mult * 0.0;
          GdQ(3, 3) += mult * rho;
          GdQ(4, 3) += mult * rho * w;

          // Fifth column
          GdQ(0, 4) += mult * rho_T;
          GdQ(1, 4) += mult * rho_T * u;
          GdQ(2, 4) += mult * rho_T * v;
          GdQ(3, 4) += mult * rho_T * w;
          GdQ(4, 4) += mult * (rho_T * H + rho * cp);

          for (int n = 5; n < ne; n++) {
            // Block (2) nth column
            GdQ(0, n) += mult * rho_Y(n - 5);
            GdQ(1, n) += mult * rho_Y(n - 5) * u;
            GdQ(2, n) += mult * rho_Y(n - 5) * v;
            GdQ(3, n) += mult * rho_Y(n - 5) * w;
            double h_y = qh(i, j, k, n) - qh(i, j, k, ne);
            GdQ(4, n) += mult * (H * rho_Y(n - 5) + rho * h_y);
            // Block (3)
            GdQ(n, 0) += mult * Theta * Y(n - 5);
            GdQ(n, 1) += mult * 0.0;
            GdQ(n, 2) += mult * 0.0;
            GdQ(n, 3) += mult * 0.0;
            GdQ(n, 4) += mult * rho_T * Y(n - 5);
          }

          // Block (4)
          for (int n = 5; n < ne; n++) {
            for (int q = 5; q < ne; q++) {
              GdQ(q, n) += mult * Y(q - 5) * rho_Y(n - 5);
            }
          }
          for (int n = 5; n < ne; n++) {
            GdQ(n, n) += mult * rho;
          }
        }

        /////////////////////////////////////////////////////////////////////////////
        // Perform LU decomposition with partial pivoting
        // Routine modifies GdQ in place resulting in a
        // strictly lower triangle matrix with 1.0 along the diagonal
        // and an upper triangular matrix including the diagonal.
        /////////////////////////////////////////////////////////////////////////////

        for (int l = 0; l < ne; l++) {
          perm(l) = l;
        }

        for (int l = 0; l < ne; l++) {
          int pivotInd = 0;
          double pivot = 0.0;
          int tempInd;
          for (int m = l; m < ne; m++)
            if (abs(GdQ(m, l)) > abs(pivot)) {
              pivot = GdQ(m, l);
              pivotInd = m;
            }

          for (int p = 0; p < ne; p++) {
            tempRow(p) = GdQ(l, p);
            GdQ(l, p) = GdQ(pivotInd, p);
            GdQ(pivotInd, p) = tempRow(p);
          }

          tempInd = perm(l);
          perm(l) = perm(pivotInd);
          perm(pivotInd) = tempInd;

          for (int p = l + 1; p < ne; p++) {
            double temp;
            temp = GdQ(p, l) /= GdQ(l, l);
            for (int q = l + 1; q < ne; q++) {
              GdQ(p, q) -= temp * GdQ(l, q);
            }
          }
        }

        // Row permute dQ to match LU
        for (int l = 0; l < ne; l++) {
          tempRow(l) = dQ(i, j, k, perm(l));
        }
        for (int l = 0; l < ne; l++) {
          dQ(i, j, k, l) = tempRow(l);
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
            tempRow(l) -= GdQ(l, q) * tempRow(q);
          }
        }

        // Now solve Ux=z which is actually
        //
        // U(dq) = tempRow
        //
        // Recall we are working with primatives so we will modify the dQ view
        // in place with the resultant dq values (as x)

        for (int l = ne - 1; l > -1; l--) {
          dQ(i, j, k, l) = tempRow(l);
          for (int q = ne - 1; q > l; q--) {
            dQ(i, j, k, l) -= GdQ(l, q) * dQ(i, j, k, q);
          }
          dQ(i, j, k, l) /= GdQ(l, l);
        }
      });
}

#include "Kokkos_Core.hpp"
#include "array"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "math.h"
#include "vector"

void dQdt(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Start off dQ with real time derivative source term
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "dQdt", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.dQ(i, j, k, l) = (3.0 * b.Q(i, j, k, l) - 4.0 * b.Qn(i, j, k, l) +
                            b.Qnm1(i, j, k, l)) /
                           (2 * dt);
      });
}

void DTrk2s1(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual time rk2 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) += b.dQ(i, j, k, l) * dt;
      });
}

void DTrk2s2(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual TIme rk2 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = 0.5 * b.Q0(i, j, k, l) +
                          0.5 * (b.q(i, j, k, l) + dt * b.dQ(i, j, k, l));
      });
}

std::vector<double> residual(std::vector<block_> mb) {
  //-------------------------------------------------------------------------------------------|
  // Compute max residual for each primative
  //-------------------------------------------------------------------------------------------|

  int ne = mb[0].ne;
  double rP, rU, rV, rW, rT, rYi;
  std::vector<double> returnResid;
  for (int l = 0; l < ne; l++) {
    returnResid.push_back(0.0);
  }

  // First we will find the max L_inf residual for each p,u,v,w,T primative
  // RECALL: Q0 array is actually primatives when using dual time
  for (block_ b : mb) {

    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "residual", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &rP,
                      double &rU, double &rV, double &rW, double &rT) {
          rP = abs(b.q(i, j, k, 0) - b.Q0(i, j, k, 0));
          rU = abs(b.q(i, j, k, 1) - b.Q0(i, j, k, 1));
          rV = abs(b.q(i, j, k, 2) - b.Q0(i, j, k, 2));
          rW = abs(b.q(i, j, k, 3) - b.Q0(i, j, k, 3));
          rT = abs(b.q(i, j, k, 4) - b.Q0(i, j, k, 4));
        },
        Kokkos::Max<double>(rP), Kokkos::Max<double>(rU),
        Kokkos::Max<double>(rV), Kokkos::Max<double>(rW),
        Kokkos::Max<double>(rT));

    returnResid[0] = fmax(rP, returnResid[0]);
    returnResid[1] = fmax(rU, returnResid[1]);
    returnResid[2] = fmax(rV, returnResid[2]);
    returnResid[3] = fmax(rW, returnResid[3]);
    returnResid[4] = fmax(rT, returnResid[4]);

    for (int n = 5; n < ne; n++) {
      Kokkos::parallel_reduce(
          "residual", range_cc,
          KOKKOS_LAMBDA(const int i, const int j, const int k, double &rYi) {
            rYi = abs(b.q(i, j, k, n) - b.Q0(i, j, k, n));
          },
          Kokkos::Max<double>(rYi));
      returnResid[n] = fmax(rP, returnResid[n]);
    }
  }

  return returnResid;
}

void invertDQ(block_ b, const double dt, const double dtau) {
  //-------------------------------------------------------------------------------------------|
  // Invert \Gamma dq = dQ to solver for dqdt
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<exec_space> token;
  int numIds = token.size();
  const int ne = 2; // b.ne;
  threeDview GdQ("GdQ", numIds, ne, ne);
  twoDview tempRow("tempRow", numIds, ne);
  twoDviewInt perm("perm", numIds, ne);
#endif

#ifdef NSCOMPILE
#define GdQ(INDEX, INDEX1) GdQ[INDEX][INDEX1]
#define perm(INDEX) perm[INDEX]
#define tempRow(INDEX) tempRow[INDEX]
#define ne 2; // 5 + NS - 1
#else
#define GdQ(INDEX, INDEX1) GdQ(id, INDEX, INDEX1)
#define perm(INDEX) perm(id, INDEX)
#define tempRow(INDEX) tempRow(id, INDEX)
#endif

  Kokkos::parallel_for(
      "dq = (Gamma + dqdQ)^{-1} dQ", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {

#ifndef NSCOMPILE
        int id = token.acquire();
#endif
#ifdef NSCOMPILE
        double GdQ(ne, ne);
        int perm(ne);
        double tempRow(ne);
#endif
        // Fill in preconditioning matrix
        // GdQ(0, 0) = 0.0;
        // GdQ(0, 1) = 0.0;
        // GdQ(0, 2) = 0.0;
        // GdQ(0, 3) = 0.0;
        // GdQ(0, 4) = 0.0;
        // GdQ(1, 0) = 0.0;
        // GdQ(1, 1) = 0.0;
        // GdQ(1, 2) = 0.0;
        // GdQ(1, 3) = 0.0;
        // GdQ(1, 4) = 0.0;
        // GdQ(2, 0) = 0.0;
        // GdQ(2, 1) = 0.0;
        // GdQ(2, 2) = 0.0;
        // GdQ(2, 3) = 0.0;
        // GdQ(2, 4) = 0.0;
        // GdQ(3, 0) = 0.0;
        // GdQ(3, 1) = 0.0;
        // GdQ(3, 2) = 0.0;
        // GdQ(3, 3) = 0.0;
        // GdQ(3, 4) = 0.0;
        // GdQ(4, 0) = 0.0;
        // GdQ(4, 1) = 0.0;
        // GdQ(4, 2) = 0.0;
        // GdQ(4, 3) = 0.0;
        // GdQ(4, 4) = 0.0;
        // for (int l = 5; l < ne; l++) {
        //   GdQ(l, 0) = 0.0;
        //   GdQ(l, 1) = 0.0;
        //   GdQ(l, 2) = 0.0;
        //   GdQ(l, 3) = 0.0;
        //   GdQ(l, 4) = 0.0;
        //   GdQ(0, l) = 0.0;
        //   GdQ(1, l) = 0.0;
        //   GdQ(2, l) = 0.0;
        //   GdQ(3, l) = 0.0;
        //   GdQ(4, l) = 0.0;
        // }

        // Add the primative to conservative transormation
        // double dotau = dt / dtau;
        // GdQ(0, 0) += 0.0 * dotau;
        // GdQ(0, 1) += 0.0 * dotau;
        // GdQ(0, 2) += 0.0 * dotau;
        // GdQ(0, 3) += 0.0 * dotau;
        // GdQ(0, 4) += 0.0 * dotau;
        // GdQ(1, 0) += 0.0 * dotau;
        // GdQ(1, 1) += 0.0 * dotau;
        // GdQ(1, 2) += 0.0 * dotau;
        // GdQ(1, 3) += 0.0 * dotau;
        // GdQ(1, 4) += 0.0 * dotau;
        // GdQ(2, 0) += 0.0 * dotau;
        // GdQ(2, 1) += 0.0 * dotau;
        // GdQ(2, 2) += 0.0 * dotau;
        // GdQ(2, 3) += 0.0 * dotau;
        // GdQ(2, 4) += 0.0 * dotau;
        // GdQ(3, 0) += 0.0 * dotau;
        // GdQ(3, 1) += 0.0 * dotau;
        // GdQ(3, 2) += 0.0 * dotau;
        // GdQ(3, 3) += 0.0 * dotau;
        // GdQ(3, 4) += 0.0 * dotau;
        // GdQ(4, 0) += 0.0 * dotau;
        // GdQ(4, 1) += 0.0 * dotau;
        // GdQ(4, 2) += 0.0 * dotau;
        // GdQ(4, 3) += 0.0 * dotau;
        // GdQ(4, 4) += 0.0 * dotau;
        // for (int l = 5; l < ne; l++) {
        //   GdQ(l, 0) += 0.0 * dotau;
        //   GdQ(l, 1) += 0.0 * dotau;
        //   GdQ(l, 2) += 0.0 * dotau;
        //   GdQ(l, 3) += 0.0 * dotau;
        //   GdQ(l, 4) += 0.0 * dotau;
        //   GdQ(0, l) += 0.0 * dotau;
        //   GdQ(1, l) += 0.0 * dotau;
        //   GdQ(2, l) += 0.0 * dotau;
        //   GdQ(3, l) += 0.0 * dotau;
        //   GdQ(4, l) += 0.0 * dotau;
        // }

        // Perform LU decomposition with partial pivoting
        // Routine modified GdQ in place resulting in a
        // strictly lower triangle matrix with 1.0 along the diagonal
        // and an upper triangular matrix including the diagonal.

        ///////////////////////////////////////////////////////////////////////////
        GdQ(0, 0) = 1.0;
        GdQ(0, 1) = 2.0;
        // GdQ(0, 2) = 7.0;
        GdQ(1, 0) = 3.0;
        GdQ(1, 1) = 5.0;
        // GdQ(1, 2) = 4.0;
        // GdQ(2, 0) = 3.0;
        // GdQ(2, 1) = 2.0;
        // GdQ(2, 2) = 1.0;
        ///////////////////////////////////////////////////////////////////////////

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
        /////////////////////////////////////////////////////////////////////////////
        printf("LU Decomposition:\n");
        for (int q = 0; q < ne; q++) {
          for (int l = 0; l < ne; l++) {
            printf("%lf ", GdQ(q, l));
          }
          printf("\n");
        }
        printf("Permutation:\n");
        for (int q = 0; q < ne; q++) {
          printf("%i ", perm(q));
        }
        printf("\n");
        /////////////////////////////////////////////////////////////////////////////

        // Solve Ax = b where A = LU by first solving for
        //
        // Lz = a then Ux=z
        //
        // Form of the equations is actually
        //
        // LU(dq) = dQ
        //
        // So begin with Lz = dQ where tempRow = z
        // Because L(0,0) == 1, we can just set the first element of z

        tempRow(0) = b.dQ(i, j, k, perm(0));
        for (int l = 1; l < ne; l++) {
          tempRow(l) = b.dQ(i, j, k, perm(l));
          for (int q = 0; q < l; q++) {
            tempRow(l) -= GdQ(l, q) * tempRow(l);
          }
        }

        // Now solve Ux=z which is actually
        //
        // U(dq) = tempRow
        //
        // Recall we are working with primatives so we will modify the dQ view
        // in place with the resultant dq values

        for (int l = ne - 1; l > 0; l--) {
          b.dQ(i, j, k, perm(l)) = tempRow(l);
          for (int q = ne - 1; q > l; q--) {
            b.dQ(i, j, k, perm(l)) -= GdQ(l, q) * tempRow(l);
          }
          b.dQ(i, j, k, perm(l)) /= GdQ(l, l);
        }

        printf("x:\n");
        for (int l = 0; l < ne; l++) {
          printf("%f ", b.dQ(i, j, k, l));
        }
        printf("\n");
      });
}

// ****************************************************************************
//
//    A 32-Species Skeletal Mechanism for C2H4-air
//
//    Z. Luo, C.S. Yoo, E.S. Richardson, J.H. Chen, C.K. Law, and T.F. Lu,
//    "Chemical explosive mode analysis for a turbulent lifted ethylene jet
//    flame in highly-heated coflow,"
//    Combustion and Flame,
//    doi:10.1016/j.combustflame.2011.05.023, 2011.
//
// ****************************************************************************
// ========================================================== //
// Y(  0) = H2
// Y(  1) = H
// Y(  2) = O
// Y(  3) = O2
// Y(  4) = OH
// Y(  5) = H2O
// Y(  6) = HO2
// Y(  7) = H2O2
// Y(  8) = CH
// Y(  9) = CH2
// Y( 10) = CH2*
// Y( 11) = CH3
// Y( 12) = CH4
// Y( 13) = CO
// Y( 14) = CO2
// Y( 15) = HCO
// Y( 16) = CH2O
// Y( 17) = CH3O
// Y( 18) = C2H2
// Y( 19) = H2CC
// Y( 20) = C2H3
// Y( 21) = C2H4
// Y( 22) = C2H5
// Y( 23) = C2H6
// Y( 24) = HCCO
// Y( 25) = CH2CO
// Y( 26) = CH2CHO
// Y( 27) = CH3CHO
// Y( 28) = aC3H5
// Y( 29) = C3H6
// Y( 30) = nC3H7
// Y( 31) = N2

// 206 reactions.
// ========================================================== //

#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>

void chem_C2H4_Air_Skeletal(block_ &b, const thtrdat_ &th,
                            const int &rface /*=0*/, const int &indxI /*=0*/,
                            const int &indxJ /*=0*/, const int &indxK /*=0*/,
                            const int &nChemSubSteps /*=1*/,
                            const double &dt /*=1.0*/) {

  // --------------------------------------------------------------|
  // cc range
  // --------------------------------------------------------------|
  MDRange3 range = getRange3(b, rface, indxI, indxJ, indxK);

  Kokkos::parallel_for(
      "Compute chemical source terms", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double T = b.q(i, j, k, 4);
        double &rho = b.Q(i, j, k, 0);
        double Y[32];
        double dYdt[32];
        double dTdt = 0.0;
        double tSub = dt / nChemSubSteps;

        // Set the initial values of Y array
        for (int n = 0; n < 31; n++) {
          Y[n] = b.q(i, j, k, 5 + n);
        }

        for (int nSub = 0; nSub < nChemSubSteps; nSub++) {

          double logT = log(T);
          double prefRuT = 101325.0 / (th.Ru * T);

          // Compute nth species Y
          Y[31] = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < 31; n++) {
            Y[n] = fmax(fmin(Y[n], 1.0), 0.0);
            Y[31] -= Y[n];
            testSum += Y[n];
          }
          if (testSum > 1.0) {
            Y[31] = 0.0;
            for (int n = 0; n < 31; n++) {
              Y[n] /= testSum;
            }
          }

          // Concentrations
          double cs[32];
          for (int n = 0; n <= 31; n++) {
            cs[n] = rho * Y[n] / th.MW(n);
          }

          // ----------------------------------------------------------- >
          // Chaperon efficiencies. ------------------------------------ >
          // ----------------------------------------------------------- >

          double cTBC[35];

          cTBC[0] = cs[1] + cs[2] + cs[3] + cs[4] + cs[6] + cs[7] + cs[8] +
                    cs[9] + cs[10] + cs[11] + 2.0 * cs[12] + cs[13] + cs[15] +
                    cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                    3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                    cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[1] = cs[0];

          cTBC[2] = cs[5];

          cTBC[3] = cs[14];

          cTBC[4] = 0.73 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] +
                    3.65 * cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] +
                    cs[11] + 2.0 * cs[12] + cs[13] + cs[14] + cs[15] + cs[16] +
                    cs[17] + 3.0 * cs[18] + cs[19] + cs[20] + 3.0 * cs[21] +
                    cs[22] + 3.0 * cs[23] + cs[24] + cs[25] + cs[26] + cs[27] +
                    cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[5] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                    cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                    2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                    cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                    3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                    cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[6] = 2.4 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 15.4 * cs[5] +
                    cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                    2.0 * cs[12] + 1.75 * cs[13] + 3.6 * cs[14] + cs[15] +
                    cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                    3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                    cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[7] = cs[0] + cs[1] + cs[2] + cs[4] + cs[6] + cs[7] + cs[8] +
                    cs[9] + cs[10] + cs[11] + cs[12] + 0.75 * cs[13] +
                    1.5 * cs[14] + cs[15] + cs[16] + cs[17] + 3.0 * cs[18] +
                    cs[19] + cs[20] + 3.0 * cs[21] + cs[22] + 1.5 * cs[23] +
                    cs[24] + cs[25] + cs[26] + cs[27] + cs[28] + cs[29] +
                    cs[30];

          cTBC[8] = cs[3];

          cTBC[9] = cs[5];

          cTBC[10] = cs[31];

          cTBC[11] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[12] = 2.0 * cs[0] + cs[1] + cs[2] + 6.0 * cs[3] + cs[4] +
                     6.0 * cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] +
                     cs[11] + 2.0 * cs[12] + 1.5 * cs[13] + 3.5 * cs[14] +
                     cs[15] + cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[13] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[14] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[15] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[16] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[17] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[18] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[19] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[20] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[21] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[22] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[23] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 2.5 * cs[18] + cs[19] + cs[20] +
                     2.5 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[24] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[25] = cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + cs[6] +
                     cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + cs[13] +
                     cs[14] + cs[15] + cs[16] + cs[17] + cs[18] + cs[19] +
                     cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[26] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[27] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[28] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[29] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[30] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[31] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[32] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[33] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + 3.0 * cs[18] + cs[19] + cs[20] +
                     3.0 * cs[21] + cs[22] + 3.0 * cs[23] + cs[24] + cs[25] +
                     cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31];

          cTBC[34] = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0 * cs[5] +
                     cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] +
                     2.0 * cs[12] + 1.5 * cs[13] + 2.0 * cs[14] + cs[15] +
                     cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] +
                     cs[22] + 3.0 * cs[23] + cs[24] + cs[25] + cs[26] + cs[27] +
                     cs[28] + cs[29] + cs[30] + cs[31];

          // ----------------------------------------------------------- >
          // Gibbs energy. --------------------------------------------- >
          // ----------------------------------------------------------- >

          int m;
          double hi[32];
          double gbs[32];
          double cp = 0.0;

          for (int n = 0; n <= 31; n++) {
            m = (T <= th.NASA7(n, 0)) ? 8 : 1;
            double cps = (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T +
                          th.NASA7(n, m + 2) * pow(T, 2.0) +
                          th.NASA7(n, m + 3) * pow(T, 3.0) +
                          th.NASA7(n, m + 4) * pow(T, 4.0)) *
                         th.Ru / th.MW(n);

            hi[n] = th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T / 2.0 +
                    th.NASA7(n, m + 2) * pow(T, 2.0) / 3.0 +
                    th.NASA7(n, m + 3) * pow(T, 3.0) / 4.0 +
                    th.NASA7(n, m + 4) * pow(T, 4.0) / 5.0 +
                    th.NASA7(n, m + 5) / T;

            double scs = th.NASA7(n, m + 0) * log(T) + th.NASA7(n, m + 1) * T +
                         th.NASA7(n, m + 2) * pow(T, 2.0) / 2.0 +
                         th.NASA7(n, m + 3) * pow(T, 3.0) / 3.0 +
                         th.NASA7(n, m + 4) * pow(T, 4.0) / 4.0 +
                         th.NASA7(n, m + 6);

            cp += cps * Y[n];
            gbs[n] = hi[n] - scs;
          }

          // ----------------------------------------------------------- >
          // Rate Constants. ------------------------------------------- >
          // FallOff Modifications. ------------------------------------ >
          // Forward, backward, net rates of progress. ----------------- >
          // ----------------------------------------------------------- >

          double k_f, dG, K_c;

          double Fcent;
          double pmod;
          double Pr, k0;
          double A, f1, F_pdr;
          double C, N;

          double q_f, q_b;
          double q[206];

          // Reaction #0
          k_f = exp(log(83000000000.00002) - (7252.903136317711 / T));
          dG = -gbs[1] + gbs[2] - gbs[3] + gbs[4];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[3];
          q_b = -k_f / K_c * cs[2] * cs[4];
          q[0] = q_f + q_b;

          // Reaction #1
          k_f = exp(log(50.00000000000001) + 2.67 * logT -
                    (3165.2508657072367 / T));
          dG = -gbs[0] + gbs[1] - gbs[2] + gbs[4];
          K_c = exp(-dG);
          q_f = k_f * cs[0] * cs[2];
          q_b = -k_f / K_c * cs[1] * cs[4];
          q[1] = q_f + q_b;

          // Reaction #2
          k_f = exp(log(216000.00000000003) + 1.51 * logT -
                    (1726.0429999007665 / T));
          dG = -gbs[0] + gbs[1] - gbs[4] + gbs[5];
          K_c = exp(-dG);
          q_f = k_f * cs[0] * cs[4];
          q_b = -k_f / K_c * cs[1] * cs[5];
          q[2] = q_f + q_b;

          // Reaction #3
          k_f = exp(log(35.7) + 2.4 * logT - (-1061.7932156823956 / T));
          dG = gbs[2] - 2.0 * gbs[4] + gbs[5];
          K_c = exp(-dG);
          q_f = k_f * pow(cs[4], 2.0);
          q_b = -k_f / K_c * cs[2] * cs[5];
          q[3] = q_f + q_b;

          // Reaction #4
          k_f = exp(log(1000000000000.0002) - 1.0 * logT);
          dG = gbs[0] - 2.0 * gbs[1];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #4
          k_f *= cTBC[0];
          q_f = k_f * pow(cs[1], 2.0);
          q_b = -k_f / K_c * cs[0];
          q[4] = q_f + q_b;

          // Reaction #5
          k_f = exp(log(90000000000.00002) - 0.6 * logT);
          dG = gbs[0] - 2.0 * gbs[1];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #5
          k_f *= cTBC[1];
          q_f = k_f * pow(cs[1], 2.0);
          q_b = -k_f / K_c * cs[0];
          q[5] = q_f + q_b;

          // Reaction #6
          k_f = exp(log(60000000000000.01) - 1.25 * logT);
          dG = gbs[0] - 2.0 * gbs[1];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #6
          k_f *= cTBC[2];
          q_f = k_f * pow(cs[1], 2.0);
          q_b = -k_f / K_c * cs[0];
          q[6] = q_f + q_b;

          // Reaction #7
          k_f = exp(log(550000000000000.1) - 2.0 * logT);
          dG = gbs[0] - 2.0 * gbs[1];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #7
          k_f *= cTBC[3];
          q_f = k_f * pow(cs[1], 2.0);
          q_b = -k_f / K_c * cs[0];
          q[7] = q_f + q_b;

          // Reaction #8
          k_f = exp(log(2.2000000000000004e+16) - 2.0 * logT);
          dG = -gbs[1] - gbs[4] + gbs[5];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #8
          k_f *= cTBC[4];
          q_f = k_f * cs[1] * cs[4];
          q_b = -k_f / K_c * cs[5];
          q[8] = q_f + q_b;

          // Reaction #9
          k_f = exp(log(500000000000.0001) - 1.0 * logT);
          dG = -gbs[1] - gbs[2] + gbs[4];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #9
          k_f *= cTBC[5];
          q_f = k_f * cs[1] * cs[2];
          q_b = -k_f / K_c * cs[4];
          q[9] = q_f + q_b;

          // Reaction #10
          k_f = exp(log(120000000000.00002) - 1.0 * logT);
          dG = -2.0 * gbs[2] + gbs[3];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #10
          k_f *= cTBC[6];
          q_f = k_f * pow(cs[2], 2.0);
          q_b = -k_f / K_c * cs[3];
          q[10] = q_f + q_b;

          // Reaction #11
          k_f = exp(log(2800000000000.0005) - 0.86 * logT);
          dG = -gbs[1] - gbs[3] + gbs[6];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #11
          k_f *= cTBC[7];
          q_f = k_f * cs[1] * cs[3];
          q_b = -k_f / K_c * cs[6];
          q[11] = q_f + q_b;

          // Reaction #12
          k_f = exp(log(300000000000000.06) - 1.72 * logT);
          dG = -gbs[1] - gbs[3] + gbs[6];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #12
          k_f *= cTBC[8];
          q_f = k_f * cs[1] * cs[3];
          q_b = -k_f / K_c * cs[6];
          q[12] = q_f + q_b;

          // Reaction #13
          k_f = exp(log(16520000000000.004) - 0.76 * logT);
          dG = -gbs[1] - gbs[3] + gbs[6];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #13
          k_f *= cTBC[9];
          q_f = k_f * cs[1] * cs[3];
          q_b = -k_f / K_c * cs[6];
          q[13] = q_f + q_b;

          // Reaction #14
          k_f = exp(log(26000000000000.004) - 1.24 * logT);
          dG = -gbs[1] - gbs[3] + gbs[6];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #14
          k_f *= cTBC[10];
          q_f = k_f * cs[1] * cs[3];
          q_b = -k_f / K_c * cs[6];
          q[14] = q_f + q_b;

          // Reaction #15
          k_f = exp(log(74000000000.00002) - 0.37 * logT);
          dG = -2.0 * gbs[4] + gbs[7];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #15
          Fcent = (1.0 - (0.7346)) * exp(-T / (94.0)) +
                  (0.7346) * exp(-T / (1756.0)) + exp(-(5182.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2300000000000.0005) - 0.9 * logT -
                   (-855.4732069479018 / T));
          Pr = cTBC[11] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * pow(cs[4], 2.0);
          q_b = -k_f / K_c * cs[7];
          q[15] = q_f + q_b;

          // Reaction #16
          k_f = exp(log(3970000000.0000005) - (337.66030697767184 / T));
          dG = -gbs[1] + gbs[2] + gbs[5] - gbs[6];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[6];
          q_b = -k_f / K_c * cs[2] * cs[5];
          q[16] = q_f + q_b;

          // Reaction #17
          k_f = exp(log(16600000000.000002) - (412.6400174689879 / T));
          dG = gbs[0] - gbs[1] + gbs[3] - gbs[6];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[6];
          q_b = -k_f / K_c * cs[0] * cs[3];
          q[17] = q_f + q_b;

          // Reaction #18
          k_f = exp(log(70800000000.00002) - (150.96586004962973 / T));
          dG = -gbs[1] + 2.0 * gbs[4] - gbs[6];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[6];
          q_b = -k_f / K_c * pow(cs[4], 2.0);
          q[18] = q_f + q_b;

          // Reaction #19
          k_f = 20000000000.000004;
          dG = -gbs[2] + gbs[3] + gbs[4] - gbs[6];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[6];
          q_b = -k_f / K_c * cs[3] * cs[4];
          q[19] = q_f + q_b;

          // Reaction #20
          k_f = exp(log(46400000000.00001) - (-251.60976674938286 / T));
          dG = gbs[3] - gbs[4] + gbs[5] - gbs[6];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[6];
          q_b = -k_f / K_c * cs[3] * cs[5];
          q[20] = q_f + q_b;

          // Reaction #21
          k_f = exp(log(130000000.00000001) - (-820.2478396029882 / T));
          dG = gbs[3] - 2.0 * gbs[6] + gbs[7];
          K_c = exp(-dG);
          q_f = k_f * pow(cs[6], 2.0);
          q_b = -k_f / K_c * cs[3] * cs[7];
          q[21] = q_f + q_b;

          // Reaction #22
          k_f = exp(log(420000000000.00006) - (6038.634401985189 / T));
          dG = gbs[3] - 2.0 * gbs[6] + gbs[7];
          K_c = exp(-dG);
          q_f = k_f * pow(cs[6], 2.0);
          q_b = -k_f / K_c * cs[3] * cs[7];
          q[22] = q_f + q_b;

          // Reaction #23
          k_f =
              exp(log(12100.000000000002) + 2 * logT - (2616.741574193582 / T));
          dG = gbs[0] - gbs[1] + gbs[6] - gbs[7];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[7];
          q_b = -k_f / K_c * cs[0] * cs[6];
          q[23] = q_f + q_b;

          // Reaction #24
          k_f = exp(log(10000000000.000002) - (1811.5903205955567 / T));
          dG = -gbs[1] + gbs[4] + gbs[5] - gbs[7];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[7];
          q_b = -k_f / K_c * cs[4] * cs[5];
          q[24] = q_f + q_b;

          // Reaction #25
          k_f = exp(log(9630.0) + 2 * logT - (2012.8781339950629 / T));
          dG = -gbs[2] + gbs[4] + gbs[6] - gbs[7];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[7];
          q_b = -k_f / K_c * cs[4] * cs[6];
          q[25] = q_f + q_b;

          // Reaction #26
          k_f = exp(log(1750000000.0000002) - (161.03025071960505 / T));
          dG = -gbs[4] + gbs[5] + gbs[6] - gbs[7];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[7];
          q_b = -k_f / K_c * cs[5] * cs[6];
          q[26] = q_f + q_b;

          // Reaction #27
          k_f = exp(log(580000000000.0001) - (4810.778740248201 / T));
          dG = -gbs[4] + gbs[5] + gbs[6] - gbs[7];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[7];
          q_b = -k_f / K_c * cs[5] * cs[6];
          q[27] = q_f + q_b;

          // Reaction #28
          k_f = exp(log(602000000.0000001) - (1509.6586004962971 / T));
          dG = -gbs[2] - gbs[13] + gbs[14];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #28
          k_f *= cTBC[12];
          q_f = k_f * cs[2] * cs[13];
          q_b = -k_f / K_c * cs[14];
          q[28] = q_f + q_b;

          // Reaction #29
          k_f = exp(log(47600.00000000001) + 1.228 * logT -
                    (35.2253673449136 / T));
          dG = gbs[1] - gbs[4] - gbs[13] + gbs[14];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[13];
          q_b = -k_f / K_c * cs[1] * cs[14];
          q[29] = q_f + q_b;

          // Reaction #30
          k_f = exp(log(43000.00000000001) + 1.5 * logT -
                    (40056.27486650175 / T));
          dG = -gbs[0] - gbs[13] + gbs[16];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #30
          Fcent = (1.0 - (0.932)) * exp(-T / (197.00000000000003)) +
                  (0.932) * exp(-T / (1540.0)) + exp(-(10300.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(5.07e+21) - 3.42 * logT - (42446.56765062089 / T));
          Pr = cTBC[13] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[0] * cs[13];
          q_b = -k_f / K_c * cs[16];
          q[30] = q_f + q_b;

          // Reaction #31
          k_f = exp(log(2500000000.0000005) - (24053.893701241002 / T));
          dG = gbs[2] - gbs[3] - gbs[13] + gbs[14];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[13];
          q_b = -k_f / K_c * cs[2] * cs[14];
          q[31] = q_f + q_b;

          // Reaction #32
          k_f = exp(log(150000000000.00003) - (11875.980990570872 / T));
          dG = gbs[4] - gbs[6] - gbs[13] + gbs[14];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[13];
          q_b = -k_f / K_c * cs[4] * cs[14];
          q[32] = q_f + q_b;

          // Reaction #33
          k_f = 57000000000.00001;
          dG = gbs[1] - gbs[2] - gbs[8] + gbs[13];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[8];
          q_b = -k_f / K_c * cs[1] * cs[13];
          q[33] = q_f + q_b;

          // Reaction #34
          k_f = 30000000000.000004;
          dG = gbs[1] - gbs[4] - gbs[8] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[8];
          q_b = -k_f / K_c * cs[1] * cs[15];
          q[34] = q_f + q_b;

          // Reaction #35
          k_f = exp(log(110700.00000000001) + 1.79 * logT -
                    (840.3766209429388 / T));
          dG = -gbs[0] + gbs[1] - gbs[8] + gbs[9];
          K_c = exp(-dG);
          q_f = k_f * cs[0] * cs[8];
          q_b = -k_f / K_c * cs[1] * cs[9];
          q[35] = q_f + q_b;

          // Reaction #36
          k_f = exp(log(5710000000.000001) - (-379.93074779156814 / T));
          dG = gbs[1] - gbs[5] - gbs[8] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[5] * cs[8];
          q_b = -k_f / K_c * cs[1] * cs[16];
          q[36] = q_f + q_b;

          // Reaction #37
          k_f = 33000000000.000004;
          dG = gbs[2] - gbs[3] - gbs[8] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[8];
          q_b = -k_f / K_c * cs[2] * cs[15];
          q[37] = q_f + q_b;

          // Reaction #38
          k_f = 50000000000.00001;
          dG = -gbs[8] - gbs[13] + gbs[24];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #38
          Fcent = (1.0 - (0.5757)) * exp(-T / (237.00000000000003)) +
                  (0.5757) * exp(-T / (1652.0)) + exp(-(5069.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.6900000000000003e+22) - 3.74 * logT -
                   (974.2330168536105 / T));
          Pr = cTBC[14] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[8] * cs[13];
          q_b = -k_f / K_c * cs[24];
          q[38] = q_f + q_b;

          // Reaction #39
          k_f = exp(log(3400000000.0000005) - (347.22147811414834 / T));
          dG = -gbs[8] + gbs[13] - gbs[14] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[14];
          q_b = -k_f / K_c * cs[13] * cs[15];
          q[39] = q_f + q_b;

          // Reaction #40
          k_f = exp(log(1090000000.0000002) + 0.48 * logT -
                    (-130.8370787096791 / T));
          dG = -gbs[1] - gbs[15] + gbs[16];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #40
          Fcent = (1.0 - (0.7824)) * exp(-T / (271.0)) +
                  (0.7824) * exp(-T / (2755.0)) + exp(-(6570.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.3500000000000003e+18) - 2.57 * logT -
                   (717.0878352357412 / T));
          Pr = cTBC[15] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[15];
          q_b = -k_f / K_c * cs[16];
          q[40] = q_f + q_b;

          // Reaction #41
          k_f = 73400000000.00002;
          dG = gbs[0] - gbs[1] + gbs[13] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[15];
          q_b = -k_f / K_c * cs[0] * cs[13];
          q[41] = q_f + q_b;

          // Reaction #42
          k_f = 30000000000.000004;
          dG = -gbs[2] + gbs[4] + gbs[13] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[15];
          q_b = -k_f / K_c * cs[4] * cs[13];
          q[42] = q_f + q_b;

          // Reaction #43
          k_f = 30000000000.000004;
          dG = gbs[1] - gbs[2] + gbs[14] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[15];
          q_b = -k_f / K_c * cs[1] * cs[14];
          q[43] = q_f + q_b;

          // Reaction #44
          k_f = 50000000000.00001;
          dG = -gbs[4] + gbs[5] + gbs[13] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[15];
          q_b = -k_f / K_c * cs[5] * cs[13];
          q[44] = q_f + q_b;

          // Reaction #45
          k_f = exp(log(187000000000000.03) - 1.0 * logT -
                    (8554.732069479018 / T));
          dG = gbs[1] + gbs[13] - gbs[15];
          K_c = prefRuT * exp(-dG);
          //  Three Body Reaction #45
          k_f *= cTBC[16];
          q_f = k_f * cs[15];
          q_b = -k_f / K_c * cs[1] * cs[13];
          q[45] = q_f + q_b;

          // Reaction #46
          k_f = exp(log(7600000000.000001) - (201.2878133995063 / T));
          dG = -gbs[3] + gbs[6] + gbs[13] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[15];
          q_b = -k_f / K_c * cs[6] * cs[13];
          q[46] = q_f + q_b;

          // Reaction #47
          k_f = exp(log(25000000000000.004) - 0.8 * logT);
          dG = -gbs[1] - gbs[9] + gbs[11];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #47
          Fcent = (1.0 - (0.68)) * exp(-T / (78.0)) +
                  (0.68) * exp(-T / (1995.0)) + exp(-(5590.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(3.2000000000000005e+21) - 3.14 * logT -
                   (618.9600262034819 / T));
          Pr = cTBC[17] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[9];
          q_b = -k_f / K_c * cs[11];
          q[47] = q_f + q_b;

          // Reaction #48
          k_f =
              exp(log(500.0000000000001) + 2 * logT - (3638.277227196076 / T));
          dG = -gbs[0] + gbs[1] - gbs[9] + gbs[11];
          K_c = exp(-dG);
          q_f = k_f * cs[0] * cs[9];
          q_b = -k_f / K_c * cs[1] * cs[11];
          q[48] = q_f + q_b;

          // Reaction #49
          k_f = 80000000000.00002;
          dG = gbs[1] - gbs[2] - gbs[9] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[9];
          q_b = -k_f / K_c * cs[1] * cs[15];
          q[49] = q_f + q_b;

          // Reaction #50
          k_f = exp(log(10560000000.000002) - (754.8293002481486 / T));
          dG = -gbs[3] + gbs[4] - gbs[9] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[9];
          q_b = -k_f / K_c * cs[4] * cs[15];
          q[50] = q_f + q_b;

          // Reaction #51
          k_f = exp(log(2640000000.0000005) - (754.8293002481486 / T));
          dG = 2.0 * gbs[1] - gbs[3] - gbs[9] + gbs[14];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[3] * cs[9];
          q_b = -k_f / K_c * pow(cs[1], 2.0) * cs[14];
          q[51] = q_f + q_b;

          // Reaction #52
          k_f = 20000000000.000004;
          dG = gbs[1] - gbs[4] - gbs[9] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[9];
          q_b = -k_f / K_c * cs[1] * cs[16];
          q[52] = q_f + q_b;

          // Reaction #53
          k_f = exp(log(11300.000000000002) + 2 * logT -
                    (1509.6586004962971 / T));
          dG = -gbs[4] + gbs[5] + gbs[8] - gbs[9];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[9];
          q_b = -k_f / K_c * cs[5] * cs[8];
          q[53] = q_f + q_b;

          // Reaction #54
          k_f = 20000000000.000004;
          dG = gbs[4] - gbs[6] - gbs[9] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[9];
          q_b = -k_f / K_c * cs[4] * cs[16];
          q[54] = q_f + q_b;

          // Reaction #55
          k_f = exp(log(810000000.0000001) + 0.5 * logT -
                    (2269.5200960794336 / T));
          dG = -gbs[9] - gbs[13] + gbs[25];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #55
          Fcent = (1.0 - (0.5907)) * exp(-T / (275.0)) +
                  (0.5907) * exp(-T / (1226.0)) + exp(-(5185.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.6900000000000006e+27) - 5.11 * logT -
                   (3570.342590173743 / T));
          Pr = cTBC[18] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[9] * cs[13];
          q_b = -k_f / K_c * cs[25];
          q[55] = q_f + q_b;

          // Reaction #56
          k_f = 40000000000.00001;
          dG = gbs[1] - gbs[8] - gbs[9] + gbs[18];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[9];
          q_b = -k_f / K_c * cs[1] * cs[18];
          q[56] = q_f + q_b;

          // Reaction #57
          k_f = 32000000000.000004;
          dG = gbs[0] - 2.0 * gbs[9] + gbs[18];
          K_c = exp(-dG);
          q_f = k_f * pow(cs[9], 2.0);
          q_b = -k_f / K_c * cs[0] * cs[18];
          q[57] = q_f + q_b;

          // Reaction #58
          k_f = exp(log(15000000000.000002) - (301.93172009925945 / T));
          dG = gbs[9] - gbs[10];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[31];
          q_b = -k_f / K_c * cs[9] * cs[31];
          q[58] = q_f + q_b;

          // Reaction #59
          k_f = 30000000000.000004;
          dG = gbs[0] - gbs[1] + gbs[8] - gbs[10];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[10];
          q_b = -k_f / K_c * cs[0] * cs[8];
          q[59] = q_f + q_b;

          // Reaction #60
          k_f = 15000000000.000002;
          dG = gbs[0] - gbs[2] - gbs[10] + gbs[13];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[10];
          q_b = -k_f / K_c * cs[0] * cs[13];
          q[60] = q_f + q_b;

          // Reaction #61
          k_f = 15000000000.000002;
          dG = gbs[1] - gbs[2] - gbs[10] + gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[10];
          q_b = -k_f / K_c * cs[1] * cs[15];
          q[61] = q_f + q_b;

          // Reaction #62
          k_f = 30000000000.000004;
          dG = gbs[1] - gbs[4] - gbs[10] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[10];
          q_b = -k_f / K_c * cs[1] * cs[16];
          q[62] = q_f + q_b;

          // Reaction #63
          k_f = 70000000000.00002;
          dG = -gbs[0] + gbs[1] - gbs[10] + gbs[11];
          K_c = exp(-dG);
          q_f = k_f * cs[0] * cs[10];
          q_b = -k_f / K_c * cs[1] * cs[11];
          q[63] = q_f + q_b;

          // Reaction #64
          k_f = 28000000000.000004;
          dG = gbs[1] - gbs[3] + gbs[4] - gbs[10] + gbs[13];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[3] * cs[10];
          q_b = -k_f / K_c * cs[1] * cs[4] * cs[13];
          q[64] = q_f + q_b;

          // Reaction #65
          k_f = 12000000000.000002;
          dG = -gbs[3] + gbs[5] - gbs[10] + gbs[13];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[10];
          q_b = -k_f / K_c * cs[5] * cs[13];
          q[65] = q_f + q_b;

          // Reaction #66
          k_f = 30000000000.000004;
          dG = gbs[9] - gbs[10];
          K_c = exp(-dG);
          q_f = k_f * cs[5] * cs[10];
          q_b = -k_f / K_c * cs[5] * cs[9];
          q[66] = q_f + q_b;

          // Reaction #67
          k_f = 9000000000.000002;
          dG = gbs[9] - gbs[10];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[13];
          q_b = -k_f / K_c * cs[9] * cs[13];
          q[67] = q_f + q_b;

          // Reaction #68
          k_f = 7000000000.000001;
          dG = gbs[9] - gbs[10];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[14];
          q_b = -k_f / K_c * cs[9] * cs[14];
          q[68] = q_f + q_b;

          // Reaction #69
          k_f = 14000000000.000002;
          dG = -gbs[10] + gbs[13] - gbs[14] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[14];
          q_b = -k_f / K_c * cs[13] * cs[16];
          q[69] = q_f + q_b;

          // Reaction #70
          k_f = exp(log(540000000.0000001) + 0.454 * logT -
                    (1308.370787096791 / T));
          dG = -gbs[1] - gbs[16] + gbs[17];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #70
          Fcent = (1.0 - (0.758)) * exp(-T / (94.0)) +
                  (0.758) * exp(-T / (1555.0)) + exp(-(4200.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.2000000000000006e+24) - 4.8 * logT -
                   (2797.9006062531375 / T));
          Pr = cTBC[19] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[16];
          q_b = -k_f / K_c * cs[17];
          q[70] = q_f + q_b;

          // Reaction #71
          k_f = exp(log(23000000.000000004) + 1.05 * logT -
                    (1648.0439722084577 / T));
          dG = gbs[0] - gbs[1] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[16];
          q_b = -k_f / K_c * cs[0] * cs[15];
          q[71] = q_f + q_b;

          // Reaction #72
          k_f = exp(log(39000000000.00001) - (1781.3971485856307 / T));
          dG = -gbs[2] + gbs[4] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[16];
          q_b = -k_f / K_c * cs[4] * cs[15];
          q[72] = q_f + q_b;

          // Reaction #73
          k_f = exp(log(3430000.0000000005) + 1.18 * logT -
                    (-224.9391314739483 / T));
          dG = -gbs[4] + gbs[5] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[16];
          q_b = -k_f / K_c * cs[5] * cs[15];
          q[73] = q_f + q_b;

          // Reaction #74
          k_f = exp(log(100000000000.00002) - (20128.78133995063 / T));
          dG = -gbs[3] + gbs[6] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[16];
          q_b = -k_f / K_c * cs[6] * cs[15];
          q[74] = q_f + q_b;

          // Reaction #75
          k_f = exp(log(1000000000.0000001) - (4025.7562679901257 / T));
          dG = -gbs[6] + gbs[7] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[16];
          q_b = -k_f / K_c * cs[7] * cs[15];
          q[75] = q_f + q_b;

          // Reaction #76
          k_f = exp(log(94600000000.00002) - (-259.15805975186436 / T));
          dG = gbs[1] - gbs[8] - gbs[16] + gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[16];
          q_b = -k_f / K_c * cs[1] * cs[25];
          q[76] = q_f + q_b;

          // Reaction #77
          k_f = exp(log(12700000000000.002) - 0.63 * logT -
                    (192.7330813300273 / T));
          dG = -gbs[1] - gbs[11] + gbs[12];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #77
          Fcent = (1.0 - (0.783)) * exp(-T / (74.0)) +
                  (0.783) * exp(-T / (2941.0)) + exp(-(6964.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.4770000000000003e+27) - 4.76 * logT -
                   (1227.8556617369884 / T));
          Pr = cTBC[20] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[11];
          q_b = -k_f / K_c * cs[12];
          q[77] = q_f + q_b;

          // Reaction #78
          k_f = 84300000000.00002;
          dG = gbs[1] - gbs[2] - gbs[11] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[11];
          q_b = -k_f / K_c * cs[1] * cs[16];
          q[78] = q_f + q_b;

          // Reaction #79
          k_f = exp(log(56000.00000000001) + 1.6 * logT -
                    (2727.4498715633104 / T));
          dG = -gbs[4] + gbs[5] + gbs[9] - gbs[11];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[11];
          q_b = -k_f / K_c * cs[5] * cs[9];
          q[79] = q_f + q_b;

          // Reaction #80
          k_f = 25010000000.000004;
          dG = -gbs[4] + gbs[5] + gbs[10] - gbs[11];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[11];
          q_b = -k_f / K_c * cs[5] * cs[10];
          q[80] = q_f + q_b;

          // Reaction #81
          k_f = exp(log(30830000000.000004) - (14492.722564764454 / T));
          dG = gbs[2] - gbs[3] - gbs[11] + gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[11];
          q_b = -k_f / K_c * cs[2] * cs[17];
          q[81] = q_f + q_b;

          // Reaction #82
          k_f = exp(log(36000000.00000001) - (4498.782629478966 / T));
          dG = -gbs[3] + gbs[4] - gbs[11] + gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[11];
          q_b = -k_f / K_c * cs[4] * cs[16];
          q[82] = q_f + q_b;

          // Reaction #83
          k_f = 1000000000.0000001;
          dG = gbs[3] - gbs[6] - gbs[11] + gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[11];
          q_b = -k_f / K_c * cs[3] * cs[12];
          q[83] = q_f + q_b;

          // Reaction #84
          k_f = 13400000000.000002;
          dG = gbs[4] - gbs[6] - gbs[11] + gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[11];
          q_b = -k_f / K_c * cs[4] * cs[17];
          q[84] = q_f + q_b;

          // Reaction #85
          k_f = exp(log(24.500000000000004) + 2.47 * logT -
                    (2606.6771835236063 / T));
          dG = gbs[6] - gbs[7] - gbs[11] + gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[7] * cs[11];
          q_b = -k_f / K_c * cs[6] * cs[12];
          q[85] = q_f + q_b;

          // Reaction #86
          k_f = 30000000000.000004;
          dG = gbs[1] - gbs[8] - gbs[11] + gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[11];
          q_b = -k_f / K_c * cs[1] * cs[20];
          q[86] = q_f + q_b;

          // Reaction #87
          k_f = 8480000000.000002;
          dG = -gbs[11] + gbs[12] + gbs[13] - gbs[15];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[15];
          q_b = -k_f / K_c * cs[12] * cs[13];
          q[87] = q_f + q_b;

          // Reaction #88
          k_f = 18000000000.000004;
          dG = -gbs[11] - gbs[15] + gbs[27];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #88
          Fcent = (1.0 - (0.6173)) * exp(-T / (13.076000000000002)) +
                  (0.6173) * exp(-T / (2078.0)) + exp(-(5093.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.2000000000000004e+42) - 9.588 * logT -
                   (2566.419620843705 / T));
          Pr = cTBC[21] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[11] * cs[15];
          q_b = -k_f / K_c * cs[27];
          q[88] = q_f + q_b;

          // Reaction #89
          k_f = exp(log(3.3200000000000003) + 2.81 * logT -
                    (2948.866466302767 / T));
          dG = -gbs[11] + gbs[12] + gbs[15] - gbs[16];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[16];
          q_b = -k_f / K_c * cs[12] * cs[15];
          q[89] = q_f + q_b;

          // Reaction #90
          k_f = 40000000000.00001;
          dG = gbs[1] - gbs[9] - gbs[11] + gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[9] * cs[11];
          q_b = -k_f / K_c * cs[1] * cs[21];
          q[90] = q_f + q_b;

          // Reaction #91
          k_f = exp(log(12000000000.000002) - (-286.83513409429645 / T));
          dG = gbs[1] - gbs[10] - gbs[11] + gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[11];
          q_b = -k_f / K_c * cs[1] * cs[21];
          q[91] = q_f + q_b;

          // Reaction #92
          k_f = exp(log(21200000000000.004) - 0.97 * logT -
                    (311.99611076923475 / T));
          dG = -2.0 * gbs[11] + gbs[23];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #92
          Fcent = (1.0 - (0.5325)) * exp(-T / (151.0)) +
                  (0.5325) * exp(-T / (1038.0)) + exp(-(4970.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.7700000000000004e+44) - 9.67 * logT -
                   (3130.025498362323 / T));
          Pr = cTBC[22] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * pow(cs[11], 2.0);
          q_b = -k_f / K_c * cs[23];
          q[92] = q_f + q_b;

          // Reaction #93
          k_f = exp(log(4990000000.000001) + 0.1 * logT -
                    (5334.127055086917 / T));
          dG = gbs[1] - 2.0 * gbs[11] + gbs[22];
          K_c = exp(-dG);
          q_f = k_f * pow(cs[11], 2.0);
          q_b = -k_f / K_c * cs[1] * cs[22];
          q[93] = q_f + q_b;

          // Reaction #94
          k_f = 50000000000.00001;
          dG = -gbs[11] + gbs[13] + gbs[21] - gbs[24];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[24];
          q_b = -k_f / K_c * cs[13] * cs[21];
          q[94] = q_f + q_b;

          // Reaction #95
          k_f = 20000000000.000004;
          dG = gbs[0] - gbs[1] + gbs[16] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[17];
          q_b = -k_f / K_c * cs[0] * cs[16];
          q[95] = q_f + q_b;

          // Reaction #96
          k_f = 32000000000.000004;
          dG = -gbs[1] + gbs[4] + gbs[11] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[17];
          q_b = -k_f / K_c * cs[4] * cs[11];
          q[96] = q_f + q_b;

          // Reaction #97
          k_f = 16000000000.000002;
          dG = -gbs[1] + gbs[5] + gbs[10] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[17];
          q_b = -k_f / K_c * cs[5] * cs[10];
          q[97] = q_f + q_b;

          // Reaction #98
          k_f = 10000000000.000002;
          dG = -gbs[2] + gbs[4] + gbs[16] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[17];
          q_b = -k_f / K_c * cs[4] * cs[16];
          q[98] = q_f + q_b;

          // Reaction #99
          k_f = 5000000000.000001;
          dG = -gbs[4] + gbs[5] + gbs[16] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[17];
          q_b = -k_f / K_c * cs[5] * cs[16];
          q[99] = q_f + q_b;

          // Reaction #100
          k_f = exp(log(4.2800000000000005e-16) + 7.6 * logT -
                    (-1776.364953250643 / T));
          dG = -gbs[3] + gbs[6] + gbs[16] - gbs[17];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[17];
          q_b = -k_f / K_c * cs[6] * cs[16];
          q[100] = q_f + q_b;

          // Reaction #101
          k_f = exp(log(660000.0000000001) + 1.62 * logT -
                    (5454.899743126621 / T));
          dG = gbs[0] - gbs[1] + gbs[11] - gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[12];
          q_b = -k_f / K_c * cs[0] * cs[11];
          q[101] = q_f + q_b;

          // Reaction #102
          k_f = exp(log(1020000.0000000001) + 1.5 * logT -
                    (4327.687988089386 / T));
          dG = -gbs[2] + gbs[4] + gbs[11] - gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[12];
          q_b = -k_f / K_c * cs[4] * cs[11];
          q[102] = q_f + q_b;

          // Reaction #103
          k_f = exp(log(100000.00000000001) + 1.6 * logT -
                    (1570.0449445161491 / T));
          dG = -gbs[4] + gbs[5] + gbs[11] - gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[12];
          q_b = -k_f / K_c * cs[5] * cs[11];
          q[103] = q_f + q_b;

          // Reaction #104
          k_f = 60000000000.00001;
          dG = gbs[1] - gbs[8] - gbs[12] + gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[12];
          q_b = -k_f / K_c * cs[1] * cs[21];
          q[104] = q_f + q_b;

          // Reaction #105
          k_f = exp(log(2460.0000000000005) + 2 * logT -
                    (4161.6255420347925 / T));
          dG = -gbs[9] + 2.0 * gbs[11] - gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[9] * cs[12];
          q_b = -k_f / K_c * pow(cs[11], 2.0);
          q[105] = q_f + q_b;

          // Reaction #106
          k_f = exp(log(16000000000.000002) - (-286.83513409429645 / T));
          dG = -gbs[10] + 2.0 * gbs[11] - gbs[12];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[12];
          q_b = -k_f / K_c * pow(cs[11], 2.0);
          q[106] = q_f + q_b;

          // Reaction #107
          k_f = 100000000000.00002;
          dG = -gbs[1] + gbs[10] + gbs[13] - gbs[24];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[24];
          q_b = -k_f / K_c * cs[10] * cs[13];
          q[107] = q_f + q_b;

          // Reaction #108
          k_f = 100000000000.00002;
          dG = gbs[1] - gbs[2] + 2.0 * gbs[13] - gbs[24];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[2] * cs[24];
          q_b = -k_f / K_c * cs[1] * pow(cs[13], 2.0);
          q[108] = q_f + q_b;

          // Reaction #109
          k_f = exp(log(1600000000.0000002) - (429.74948160794594 / T));
          dG = -gbs[3] + gbs[4] + 2.0 * gbs[13] - gbs[24];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[3] * cs[24];
          q_b = -k_f / K_c * cs[4] * pow(cs[13], 2.0);
          q[109] = q_f + q_b;

          // Reaction #110
          k_f = 50000000000.00001;
          dG = -gbs[8] + gbs[13] + gbs[18] - gbs[24];
          K_c = exp(-dG);
          q_f = k_f * cs[8] * cs[24];
          q_b = -k_f / K_c * cs[13] * cs[18];
          q[110] = q_f + q_b;

          // Reaction #111
          k_f = 30000000000.000004;
          dG = -gbs[9] + gbs[13] + gbs[20] - gbs[24];
          K_c = exp(-dG);
          q_f = k_f * cs[9] * cs[24];
          q_b = -k_f / K_c * cs[13] * cs[20];
          q[111] = q_f + q_b;

          // Reaction #112
          k_f = 10000000000.000002;
          dG = 2.0 * gbs[13] + gbs[18] - 2.0 * gbs[24];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * pow(cs[24], 2.0);
          q_b = -k_f / K_c * pow(cs[13], 2.0) * cs[18];
          q[112] = q_f + q_b;

          // Reaction #113
          k_f = exp(log(800000000000000.0) - 0.52 * logT -
                    (25538.391325062363 / T));
          dG = -gbs[18] + gbs[19];
          K_c = exp(-dG);
          //  Lindeman Reaction #113
          Fcent = 1.0;
          k0 = exp(log(2450000000000.0005) - 0.64 * logT -
                   (25010.010814888657 / T));
          Pr = cTBC[23] * k0 / k_f;
          pmod = Pr / (1.0 + Pr);
          k_f *= pmod;
          q_f = k_f * cs[18];
          q_b = -k_f / K_c * cs[19];
          q[113] = q_f + q_b;

          // Reaction #114
          k_f = exp(log(386000000.0) + 1.62 * logT - (18643.387985359645 / T));
          dG = gbs[1] + gbs[18] - gbs[20];
          K_c = prefRuT * exp(-dG);
          //  Troe Reaction #114
          Fcent = (1.0 - (1.9816)) * exp(-T / (5383.7)) +
                  (1.9816) * exp(-T / (4.2932)) + exp(-(-0.0795) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(2.565e+24) - 3.4 * logT - (18014.615178252938 / T));
          Pr = cTBC[24] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[20];
          q_b = -k_f / K_c * cs[1] * cs[18];
          q[114] = q_f + q_b;

          // Reaction #115
          k_f =
              exp(log(16320.000000000004) + 2 * logT - (956.117113647655 / T));
          dG = gbs[1] - gbs[2] - gbs[18] + gbs[24];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[18];
          q_b = -k_f / K_c * cs[1] * cs[24];
          q[115] = q_f + q_b;

          // Reaction #116
          k_f = exp(log(4080.000000000001) + 2 * logT - (956.117113647655 / T));
          dG = -gbs[2] + gbs[9] + gbs[13] - gbs[18];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[18];
          q_b = -k_f / K_c * cs[9] * cs[13];
          q[116] = q_f + q_b;

          // Reaction #117
          k_f = exp(log(2.1800000000000005e-07) + 4.5 * logT -
                    (-503.2195334987657 / T));
          dG = gbs[1] - gbs[4] - gbs[18] + gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[18];
          q_b = -k_f / K_c * cs[1] * cs[25];
          q[117] = q_f + q_b;

          // Reaction #118
          k_f = exp(log(4.830000000000001e-07) + 4 * logT -
                    (-1006.4390669975314 / T));
          dG = -gbs[4] + gbs[11] + gbs[13] - gbs[18];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[18];
          q_b = -k_f / K_c * cs[11] * cs[13];
          q[118] = q_f + q_b;

          // Reaction #119
          k_f = exp(log(10000.000000000002) + 2 * logT -
                    (3019.3172009925943 / T));
          dG = gbs[13] - gbs[15] - gbs[18] + gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[15] * cs[18];
          q_b = -k_f / K_c * cs[13] * cs[20];
          q[119] = q_f + q_b;

          // Reaction #120
          k_f = exp(log(2.2000000000000006e+49) - 11.82 * logT -
                    (17980.0339319109 / T));
          dG = -gbs[11] - gbs[18] + gbs[28];
          K_c = exp(-dG) / prefRuT;
          //  Three Body Reaction #120
          k_f *= cTBC[25];
          q_f = k_f * cs[11] * cs[18];
          q_b = -k_f / K_c * cs[28];
          q[120] = q_f + q_b;

          // Reaction #121
          k_f = 100000000000.00002;
          dG = gbs[18] - gbs[19];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[19];
          q_b = -k_f / K_c * cs[1] * cs[18];
          q[121] = q_f + q_b;

          // Reaction #122
          k_f = 100000000000.00002;
          dG = -gbs[2] + gbs[9] + gbs[13] - gbs[19];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[19];
          q_b = -k_f / K_c * cs[9] * cs[13];
          q[122] = q_f + q_b;

          // Reaction #123
          k_f = 20000000000.000004;
          dG = gbs[1] - gbs[4] - gbs[19] + gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[19];
          q_b = -k_f / K_c * cs[1] * cs[25];
          q[123] = q_f + q_b;

          // Reaction #124
          k_f = 10000000000.000002;
          dG = -gbs[3] + gbs[9] + gbs[14] - gbs[19];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[19];
          q_b = -k_f / K_c * cs[9] * cs[14];
          q[124] = q_f + q_b;

          // Reaction #125
          k_f = exp(log(330000000000.00006) - 0.06 * logT -
                    (4277.366034739509 / T));
          dG = -gbs[1] - gbs[25] + gbs[26];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #125
          Fcent = (1.0 - (0.337)) * exp(-T / (1707.0)) +
                  (0.337) * exp(-T / (3200.0)) + exp(-(4131.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(3.8000000000000014e+35) - 7.64 * logT -
                   (5988.312448635313 / T));
          Pr = cTBC[26] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[25];
          q_b = -k_f / K_c * cs[26];
          q[125] = q_f + q_b;

          // Reaction #126
          k_f = exp(log(50000000000.00001) - (4025.7562679901257 / T));
          dG = gbs[0] - gbs[1] + gbs[24] - gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[25];
          q_b = -k_f / K_c * cs[0] * cs[24];
          q[126] = q_f + q_b;

          // Reaction #127
          k_f = exp(log(1500000.0000000002) + 1.43 * logT -
                    (1353.6605451116798 / T));
          dG = -gbs[1] + gbs[11] + gbs[13] - gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[25];
          q_b = -k_f / K_c * cs[11] * cs[13];
          q[127] = q_f + q_b;

          // Reaction #128
          k_f = exp(log(10000000000.000002) - (4025.7562679901257 / T));
          dG = -gbs[2] + gbs[4] + gbs[24] - gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[25];
          q_b = -k_f / K_c * cs[4] * cs[24];
          q[128] = q_f + q_b;

          // Reaction #129
          k_f = exp(log(1750000000.0000002) - (679.3463702233338 / T));
          dG = -gbs[2] + gbs[9] + gbs[14] - gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[25];
          q_b = -k_f / K_c * cs[9] * cs[14];
          q[129] = q_f + q_b;

          // Reaction #130
          k_f = exp(log(7500000000.000001) - (1006.4390669975314 / T));
          dG = -gbs[4] + gbs[5] + gbs[24] - gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[25];
          q_b = -k_f / K_c * cs[5] * cs[24];
          q[130] = q_f + q_b;

          // Reaction #131
          k_f = exp(log(6080000000.000001) + 0.27 * logT -
                    (140.9014693796544 / T));
          dG = -gbs[1] - gbs[20] + gbs[21];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #131
          Fcent = (1.0 - (0.782)) * exp(-T / (207.49999999999997)) +
                  (0.782) * exp(-T / (2663.0)) + exp(-(6095.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.4000000000000004e+24) - 3.86 * logT -
                   (1670.6888512159023 / T));
          Pr = cTBC[27] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[20];
          q_b = -k_f / K_c * cs[21];
          q[131] = q_f + q_b;

          // Reaction #132
          k_f = 30000000000.000004;
          dG = gbs[0] - gbs[1] + gbs[18] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[20];
          q_b = -k_f / K_c * cs[0] * cs[18];
          q[132] = q_f + q_b;

          // Reaction #133
          k_f = 60000000000.00001;
          dG = gbs[0] - gbs[1] + gbs[19] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[20];
          q_b = -k_f / K_c * cs[0] * cs[19];
          q[133] = q_f + q_b;

          // Reaction #134
          k_f = 48000000000.00001;
          dG = gbs[1] - gbs[2] - gbs[20] + gbs[25];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[20];
          q_b = -k_f / K_c * cs[1] * cs[25];
          q[134] = q_f + q_b;

          // Reaction #135
          k_f = 48000000000.00001;
          dG = -gbs[2] + gbs[11] + gbs[13] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[20];
          q_b = -k_f / K_c * cs[11] * cs[13];
          q[135] = q_f + q_b;

          // Reaction #136
          k_f = 30110000000.000004;
          dG = -gbs[4] + gbs[5] + gbs[18] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[20];
          q_b = -k_f / K_c * cs[5] * cs[18];
          q[136] = q_f + q_b;

          // Reaction #137
          k_f = exp(log(1340.0000000000002) + 1.61 * logT -
                    (-192.93436914342678 / T));
          dG = -gbs[3] + gbs[6] + gbs[18] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[20];
          q_b = -k_f / K_c * cs[6] * cs[18];
          q[137] = q_f + q_b;

          // Reaction #138
          k_f = exp(log(300000000.00000006) + 0.29 * logT -
                    (5.5354148684864235 / T));
          dG = gbs[2] - gbs[3] - gbs[20] + gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[20];
          q_b = -k_f / K_c * cs[2] * cs[26];
          q[138] = q_f + q_b;

          // Reaction #139
          k_f = exp(log(46000000000000.01) - 1.39 * logT -
                    (508.2517288337534 / T));
          dG = -gbs[3] + gbs[15] + gbs[16] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[20];
          q_b = -k_f / K_c * cs[15] * cs[16];
          q[139] = q_f + q_b;

          // Reaction #140
          k_f = 10000000000.000002;
          dG = gbs[4] - gbs[6] - gbs[20] + gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[20];
          q_b = -k_f / K_c * cs[4] * cs[26];
          q[140] = q_f + q_b;

          // Reaction #141
          k_f = exp(log(12100000.000000002) - (-299.9188419652644 / T));
          dG = gbs[6] - gbs[7] - gbs[20] + gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[7] * cs[20];
          q_b = -k_f / K_c * cs[6] * cs[21];
          q[141] = q_f + q_b;

          // Reaction #142
          k_f = 90330000000.00002;
          dG = gbs[13] - gbs[15] - gbs[20] + gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[15] * cs[20];
          q_b = -k_f / K_c * cs[13] * cs[21];
          q[142] = q_f + q_b;

          // Reaction #143
          k_f = 392000000.00000006;
          dG = -gbs[11] + gbs[12] + gbs[18] - gbs[20];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[20];
          q_b = -k_f / K_c * cs[12] * cs[18];
          q[143] = q_f + q_b;

          // Reaction #144
          k_f = 25000000000.000004;
          dG = -gbs[11] - gbs[20] + gbs[29];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #144
          Fcent = (1.0 - (0.175)) * exp(-T / (1340.6)) +
                  (0.175) * exp(-T / (60000.0)) + exp(-(10139.8) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(4.2700000000000004e+52) - 11.94 * logT -
                   (4916.354198376241 / T));
          Pr = cTBC[28] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[11] * cs[20];
          q_b = -k_f / K_c * cs[29];
          q[144] = q_f + q_b;

          // Reaction #145
          k_f = exp(log(1.5000000000000003e+21) - 2.83 * logT -
                    (9368.94127468002 / T));
          dG = gbs[1] - gbs[11] - gbs[20] + gbs[28];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[20];
          q_b = -k_f / K_c * cs[1] * cs[28];
          q[145] = q_f + q_b;

          // Reaction #146
          k_f = exp(log(7.8e+41) - 9.147 * logT - (23600.996121092114 / T));
          dG = gbs[11] + gbs[13] - gbs[26];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[26];
          q_b = -k_f / K_c * cs[11] * cs[13];
          q[146] = q_f + q_b;

          // Reaction #147
          k_f = 100000000000.00002;
          dG = -gbs[1] - gbs[26] + gbs[27];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #147
          Fcent = (1.0 - (0.55)) * exp(-T / (8900.0)) +
                  (0.55) * exp(-T / (4350.0)) + exp(-(7244.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(5.200000000000002e+33) - 7.297 * logT -
                   (2365.131807444199 / T));
          Pr = cTBC[29] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[26];
          q_b = -k_f / K_c * cs[27];
          q[147] = q_f + q_b;

          // Reaction #148
          k_f = 90000000000.00002;
          dG = -gbs[1] + gbs[11] + gbs[15] - gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[26];
          q_b = -k_f / K_c * cs[11] * cs[15];
          q[148] = q_f + q_b;

          // Reaction #149
          k_f = exp(log(20000000000.000004) - (2012.8781339950629 / T));
          dG = gbs[0] - gbs[1] + gbs[25] - gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[26];
          q_b = -k_f / K_c * cs[0] * cs[25];
          q[149] = q_f + q_b;

          // Reaction #150
          k_f = exp(log(20000000000.000004) - (2012.8781339950629 / T));
          dG = -gbs[2] + gbs[4] + gbs[25] - gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[26];
          q_b = -k_f / K_c * cs[4] * cs[25];
          q[150] = q_f + q_b;

          // Reaction #151
          k_f = exp(log(10000000000.000002) - (1006.4390669975314 / T));
          dG = -gbs[4] + gbs[5] + gbs[25] - gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[26];
          q_b = -k_f / K_c * cs[5] * cs[25];
          q[151] = q_f + q_b;

          // Reaction #152
          k_f = 140000000.00000003;
          dG = -gbs[3] + gbs[6] + gbs[25] - gbs[26];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[26];
          q_b = -k_f / K_c * cs[6] * cs[25];
          q[152] = q_f + q_b;

          // Reaction #153
          k_f = 18000000.000000004;
          dG = -gbs[3] + gbs[4] + gbs[13] + gbs[16] - gbs[26];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[3] * cs[26];
          q_b = -k_f / K_c * cs[4] * cs[13] * cs[16];
          q[153] = q_f + q_b;

          // Reaction #154
          k_f =
              exp(log(8000000000000.0) + 0.44 * logT - (44670.79798868544 / T));
          dG = gbs[0] + gbs[19] - gbs[21];
          K_c = prefRuT * exp(-dG);
          //  Troe Reaction #154
          Fcent = (1.0 - (0.7345)) * exp(-T / (180.0)) +
                  (0.7345) * exp(-T / (1035.0)) + exp(-(5417.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(7.000000000000001e+47) - 9.31 * logT -
                   (50251.50261518675 / T));
          Pr = cTBC[30] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[21];
          q_b = -k_f / K_c * cs[0] * cs[19];
          q[154] = q_f + q_b;

          // Reaction #155
          k_f = exp(log(1080000000.0000002) + 0.454 * logT -
                    (915.8595509677536 / T));
          dG = -gbs[1] - gbs[21] + gbs[22];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #155
          Fcent = (1.0 - (0.9753)) * exp(-T / (209.99999999999997)) +
                  (0.9753) * exp(-T / (983.9999999999999)) + exp(-(4374.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.2000000000000001e+36) - 7.62 * logT -
                   (3507.440148486397 / T));
          Pr = cTBC[31] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[21];
          q_b = -k_f / K_c * cs[22];
          q[155] = q_f + q_b;

          // Reaction #156
          k_f = exp(log(50700.00000000001) + 1.93 * logT -
                    (6516.692958809016 / T));
          dG = gbs[0] - gbs[1] + gbs[20] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[21];
          q_b = -k_f / K_c * cs[0] * cs[20];
          q[156] = q_f + q_b;

          // Reaction #157
          k_f = exp(log(15100.000000000004) + 1.91 * logT -
                    (1882.041055285384 / T));
          dG = -gbs[2] + gbs[4] + gbs[20] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[21];
          q_b = -k_f / K_c * cs[4] * cs[20];
          q[157] = q_f + q_b;

          // Reaction #158
          k_f = exp(log(19200.000000000004) + 1.83 * logT -
                    (110.70829736972846 / T));
          dG = -gbs[2] + gbs[11] + gbs[15] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[21];
          q_b = -k_f / K_c * cs[11] * cs[15];
          q[158] = q_f + q_b;

          // Reaction #159
          k_f = exp(log(384.00000000000006) + 1.83 * logT -
                    (110.70829736972846 / T));
          dG = -gbs[2] + gbs[9] + gbs[16] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[21];
          q_b = -k_f / K_c * cs[9] * cs[16];
          q[159] = q_f + q_b;

          // Reaction #160
          k_f = exp(log(3600.0000000000005) + 2 * logT -
                    (1258.0488337469144 / T));
          dG = -gbs[4] + gbs[5] + gbs[20] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[21];
          q_b = -k_f / K_c * cs[5] * cs[20];
          q[160] = q_f + q_b;

          // Reaction #161
          k_f = exp(log(42200000000.00001) - (30595.74763672496 / T));
          dG = -gbs[3] + gbs[6] + gbs[20] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[21];
          q_b = -k_f / K_c * cs[6] * cs[20];
          q[161] = q_f + q_b;

          // Reaction #162
          k_f = exp(log(2000000000.0000002) - (7045.07346898272 / T));
          dG = gbs[4] - gbs[6] - gbs[21] + gbs[27];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[21];
          q_b = -k_f / K_c * cs[4] * cs[27];
          q[162] = q_f + q_b;

          // Reaction #163
          k_f = exp(log(10000.000000000002) + 2 * logT -
                    (4025.7562679901257 / T));
          dG = gbs[13] - gbs[15] - gbs[21] + gbs[22];
          K_c = exp(-dG);
          q_f = k_f * cs[15] * cs[21];
          q_b = -k_f / K_c * cs[13] * cs[22];
          q[163] = q_f + q_b;

          // Reaction #164
          k_f = exp(log(20000000000.000004) - (3019.3172009925943 / T));
          dG = gbs[1] - gbs[9] - gbs[21] + gbs[28];
          K_c = exp(-dG);
          q_f = k_f * cs[9] * cs[21];
          q_b = -k_f / K_c * cs[1] * cs[28];
          q[164] = q_f + q_b;

          // Reaction #165
          k_f = 50000000000.00001;
          dG = -gbs[10] + gbs[12] + gbs[19] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[21];
          q_b = -k_f / K_c * cs[12] * cs[19];
          q[165] = q_f + q_b;

          // Reaction #166
          k_f = 50000000000.00001;
          dG = gbs[1] - gbs[10] - gbs[21] + gbs[28];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[21];
          q_b = -k_f / K_c * cs[1] * cs[28];
          q[166] = q_f + q_b;

          // Reaction #167
          k_f =
              exp(log(227.00000000000003) + 2 * logT - (4629.619708188645 / T));
          dG = -gbs[11] + gbs[12] + gbs[20] - gbs[21];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[21];
          q_b = -k_f / K_c * cs[12] * cs[20];
          q[167] = q_f + q_b;

          // Reaction #168
          k_f = exp(log(330000000.00000006) - (3874.790407940496 / T));
          dG = -gbs[11] - gbs[21] + gbs[30];
          K_c = exp(-dG) / prefRuT;
          q_f = k_f * cs[11] * cs[21];
          q_b = -k_f / K_c * cs[30];
          q[168] = q_f + q_b;

          // Reaction #169
          k_f = exp(log(521000000000000.06) - 0.99 * logT -
                    (795.0868629280499 / T));
          dG = -gbs[1] - gbs[22] + gbs[23];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #169
          Fcent = (1.0 - (0.8422)) * exp(-T / (125.0)) +
                  (0.8422) * exp(-T / (2219.0)) + exp(-(6882.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.9900000000000005e+35) - 7.08 * logT -
                   (3364.022581439249 / T));
          Pr = cTBC[32] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[22];
          q_b = -k_f / K_c * cs[23];
          q[169] = q_f + q_b;

          // Reaction #170
          k_f = 2000000000.0000002;
          dG = gbs[0] - gbs[1] + gbs[21] - gbs[22];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[22];
          q_b = -k_f / K_c * cs[0] * cs[21];
          q[170] = q_f + q_b;

          // Reaction #171
          k_f = 16040000000.000002;
          dG = -gbs[2] + gbs[11] + gbs[16] - gbs[22];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[22];
          q_b = -k_f / K_c * cs[11] * cs[16];
          q[171] = q_f + q_b;

          // Reaction #172
          k_f = 80200000000.00002;
          dG = gbs[1] - gbs[2] - gbs[22] + gbs[27];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[22];
          q_b = -k_f / K_c * cs[1] * cs[27];
          q[172] = q_f + q_b;

          // Reaction #173
          k_f = 20000000.000000004;
          dG = -gbs[3] + gbs[6] + gbs[21] - gbs[22];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[22];
          q_b = -k_f / K_c * cs[6] * cs[21];
          q[173] = q_f + q_b;

          // Reaction #174
          k_f = 300000000.00000006;
          dG = gbs[3] - gbs[6] - gbs[22] + gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[22];
          q_b = -k_f / K_c * cs[3] * cs[23];
          q[174] = q_f + q_b;

          // Reaction #175
          k_f = 300000000.00000006;
          dG = -gbs[6] + gbs[7] + gbs[21] - gbs[22];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[22];
          q_b = -k_f / K_c * cs[7] * cs[21];
          q[175] = q_f + q_b;

          // Reaction #176
          k_f = 24000000000.000004;
          dG = gbs[4] - gbs[6] + gbs[11] + gbs[16] - gbs[22];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[6] * cs[22];
          q_b = -k_f / K_c * cs[4] * cs[11] * cs[16];
          q[176] = q_f + q_b;

          // Reaction #177
          k_f = exp(log(8700000.000000002) - (490.13582562779783 / T));
          dG = gbs[6] - gbs[7] - gbs[22] + gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[7] * cs[22];
          q_b = -k_f / K_c * cs[6] * cs[23];
          q[177] = q_f + q_b;

          // Reaction #178
          k_f = 120000000000.00002;
          dG = gbs[13] - gbs[15] - gbs[22] + gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[15] * cs[22];
          q_b = -k_f / K_c * cs[13] * cs[23];
          q[178] = q_f + q_b;

          // Reaction #179
          k_f = exp(log(115000.00000000001) + 1.9 * logT -
                    (3789.243087245706 / T));
          dG = gbs[0] - gbs[1] + gbs[22] - gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[23];
          q_b = -k_f / K_c * cs[0] * cs[22];
          q[179] = q_f + q_b;

          // Reaction #180
          k_f = exp(log(89800.00000000001) + 1.92 * logT -
                    (2863.319145607977 / T));
          dG = -gbs[2] + gbs[4] + gbs[22] - gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[23];
          q_b = -k_f / K_c * cs[4] * cs[22];
          q[180] = q_f + q_b;

          // Reaction #181
          k_f = exp(log(3540.0000000000005) + 2.12 * logT -
                    (437.8009941439262 / T));
          dG = -gbs[4] + gbs[5] + gbs[22] - gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[23];
          q_b = -k_f / K_c * cs[5] * cs[22];
          q[181] = q_f + q_b;

          // Reaction #182
          k_f = exp(log(40000000000.00001) - (-276.77074342432115 / T));
          dG = -gbs[10] + gbs[11] + gbs[22] - gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[10] * cs[23];
          q_b = -k_f / K_c * cs[11] * cs[22];
          q[182] = q_f + q_b;

          // Reaction #183
          k_f = exp(log(6140.000000000002) + 1.74 * logT -
                    (5258.644125062102 / T));
          dG = -gbs[11] + gbs[12] + gbs[22] - gbs[23];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[23];
          q_b = -k_f / K_c * cs[12] * cs[22];
          q[183] = q_f + q_b;

          // Reaction #184
          k_f = 200000000000.00003;
          dG = -gbs[1] - gbs[28] + gbs[29];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #184
          Fcent = (1.0 - (0.02)) * exp(-T / (1096.6)) +
                  (0.02) * exp(-T / (1096.6)) + exp(-(6859.5) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(1.3300000000000002e+54) - 12.0 * logT -
                   (3003.113532013934 / T));
          Pr = cTBC[33] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[28];
          q_b = -k_f / K_c * cs[29];
          q[184] = q_f + q_b;

          // Reaction #185
          k_f = exp(log(20000000000.000004) - (1006.4390669975314 / T));
          dG = -gbs[1] + gbs[12] + gbs[19] - gbs[28];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[28];
          q_b = -k_f / K_c * cs[12] * cs[19];
          q[185] = q_f + q_b;

          // Reaction #186
          k_f = 2660000000.0000005;
          dG = gbs[3] - gbs[6] - gbs[28] + gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[28];
          q_b = -k_f / K_c * cs[3] * cs[29];
          q[186] = q_f + q_b;

          // Reaction #187
          k_f = 6600000000.000001;
          dG = gbs[4] - gbs[6] + gbs[16] + gbs[20] - gbs[28];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[6] * cs[28];
          q_b = -k_f / K_c * cs[4] * cs[16] * cs[20];
          q[187] = q_f + q_b;

          // Reaction #188
          k_f = 60000000000.00001;
          dG = gbs[13] - gbs[15] - gbs[28] + gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[15] * cs[28];
          q_b = -k_f / K_c * cs[13] * cs[29];
          q[188] = q_f + q_b;

          // Reaction #189
          k_f = exp(log(13300000000.000002) - (1640.8479328794253 / T));
          dG = -gbs[1] - gbs[29] + gbs[30];
          K_c = exp(-dG) / prefRuT;
          //  Troe Reaction #189
          Fcent = (1.0 - (1.0)) * exp(-T / (1000.0)) +
                  (1.0) * exp(-T / (1310.0)) + exp(-(48097.0) / T);
          C = -0.4 - 0.67 * log10(Fcent);
          N = 0.75 - 1.27 * log10(Fcent);
          k0 = exp(log(6.260000000000001e+32) - 6.66 * logT -
                   (3522.53673449136 / T));
          Pr = cTBC[34] * k0 / k_f;
          A = log10(Pr) + C;
          f1 = A / (N - 0.14 * A);
          F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
          pmod = Pr / (1.0 + Pr) * F_pdr;
          k_f *= pmod;
          q_f = k_f * cs[1] * cs[29];
          q_b = -k_f / K_c * cs[30];
          q[189] = q_f + q_b;

          // Reaction #190
          k_f = exp(log(1.6000000000000002e+19) - 2.39 * logT -
                    (5625.994384516201 / T));
          dG = -gbs[1] + gbs[11] + gbs[21] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[29];
          q_b = -k_f / K_c * cs[11] * cs[21];
          q[190] = q_f + q_b;

          // Reaction #191
          k_f = exp(log(170.00000000000003) + 2.5 * logT -
                    (1253.0166384119266 / T));
          dG = gbs[0] - gbs[1] + gbs[28] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[29];
          q_b = -k_f / K_c * cs[0] * cs[28];
          q[191] = q_f + q_b;

          // Reaction #192
          k_f = exp(log(120000.00000000001) + 1.65 * logT -
                    (164.5527874540964 / T));
          dG = gbs[1] - gbs[2] + gbs[11] + gbs[25] - gbs[29];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[2] * cs[29];
          q_b = -k_f / K_c * cs[1] * cs[11] * cs[25];
          q[192] = q_f + q_b;

          // Reaction #193
          k_f = exp(log(35000.00000000001) + 1.65 * logT -
                    (-489.1293865608003 / T));
          dG = -gbs[2] + gbs[15] + gbs[22] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[29];
          q_b = -k_f / K_c * cs[15] * cs[22];
          q[193] = q_f + q_b;

          // Reaction #194
          k_f = exp(log(180000000.00000003) + 0.7 * logT -
                    (2958.9308569727427 / T));
          dG = -gbs[2] + gbs[4] + gbs[28] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[29];
          q_b = -k_f / K_c * cs[4] * cs[28];
          q[194] = q_f + q_b;

          // Reaction #195
          k_f = exp(log(3100.0000000000005) + 2 * logT -
                    (-149.9594209826322 / T));
          dG = -gbs[4] + gbs[5] + gbs[28] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[29];
          q_b = -k_f / K_c * cs[5] * cs[28];
          q[195] = q_f + q_b;

          // Reaction #196
          k_f = exp(log(9.600000000000001) + 2.6 * logT -
                    (6999.783710967831 / T));
          dG = -gbs[6] + gbs[7] + gbs[28] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[6] * cs[29];
          q_b = -k_f / K_c * cs[7] * cs[28];
          q[196] = q_f + q_b;

          // Reaction #197
          k_f = exp(log(0.0022000000000000006) + 3.5 * logT -
                    (2855.7708526054957 / T));
          dG = -gbs[11] + gbs[12] + gbs[28] - gbs[29];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[29];
          q_b = -k_f / K_c * cs[12] * cs[28];
          q[197] = q_f + q_b;

          // Reaction #198
          k_f = exp(log(3.7000000000000005e+21) - 2.92 * logT -
                    (6292.760266402065 / T));
          dG = -gbs[1] + gbs[11] + gbs[22] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[30];
          q_b = -k_f / K_c * cs[11] * cs[22];
          q[198] = q_f + q_b;

          // Reaction #199
          k_f = 1800000000.0000002;
          dG = gbs[0] - gbs[1] + gbs[29] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[1] * cs[30];
          q_b = -k_f / K_c * cs[0] * cs[29];
          q[199] = q_f + q_b;

          // Reaction #200
          k_f = 96000000000.00002;
          dG = -gbs[2] + gbs[16] + gbs[22] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[2] * cs[30];
          q_b = -k_f / K_c * cs[16] * cs[22];
          q[200] = q_f + q_b;

          // Reaction #201
          k_f = 24000000000.000004;
          dG = -gbs[4] + gbs[5] + gbs[29] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[4] * cs[30];
          q_b = -k_f / K_c * cs[5] * cs[29];
          q[201] = q_f + q_b;

          // Reaction #202
          k_f = 90000000.00000001;
          dG = -gbs[3] + gbs[6] + gbs[29] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[3] * cs[30];
          q_b = -k_f / K_c * cs[6] * cs[29];
          q[202] = q_f + q_b;

          // Reaction #203
          k_f = 24000000000.000004;
          dG = gbs[4] - gbs[6] + gbs[16] + gbs[22] - gbs[30];
          K_c = prefRuT * exp(-dG);
          q_f = k_f * cs[6] * cs[30];
          q_b = -k_f / K_c * cs[4] * cs[16] * cs[22];
          q[203] = q_f + q_b;

          // Reaction #204
          k_f = 11000000000.000002;
          dG = -gbs[11] + gbs[12] + gbs[29] - gbs[30];
          K_c = exp(-dG);
          q_f = k_f * cs[11] * cs[30];
          q_b = -k_f / K_c * cs[12] * cs[29];
          q[204] = q_f + q_b;

          // Reaction #205
          k_f = exp(log(3.9000000000000004e+29) - 5.22 * logT -
                    (9937.076128000126 / T));
          dG = gbs[11] - gbs[20] - gbs[22] + gbs[28];
          K_c = exp(-dG);
          q_f = k_f * cs[20] * cs[22];
          q_b = -k_f / K_c * cs[11] * cs[28];
          q[205] = q_f + q_b;

          // ----------------------------------------------------------- >
          // Source terms. --------------------------------------------- >
          // ----------------------------------------------------------- >

          dYdt[0] =
              th.MW(0) *
              (-q[1] - q[2] + q[4] + q[5] + q[6] + q[7] + q[17] + q[23] -
               q[30] - q[35] + q[41] - q[48] + q[57] + q[59] + q[60] - q[63] +
               q[71] + q[95] + q[101] + q[126] + q[132] + q[133] + q[149] +
               q[154] + q[156] + q[170] + q[179] + q[191] + q[199]);
          dYdt[1] =
              th.MW(1) *
              (-q[0] + q[1] + q[2] - 2.0 * q[4] - 2.0 * q[5] - 2.0 * q[6] -
               2.0 * q[7] - q[8] - q[9] - q[11] - q[12] - q[13] - q[14] -
               q[16] - q[17] - q[18] - q[23] - q[24] + q[29] + q[33] + q[34] +
               q[35] + q[36] - q[40] - q[41] + q[43] + q[45] - q[47] + q[48] +
               q[49] + 2.0 * q[51] + q[52] + q[56] - q[59] + q[61] + q[62] +
               q[63] + q[64] - q[70] - q[71] + q[76] - q[77] + q[78] + q[86] +
               q[90] + q[91] + q[93] - q[95] - q[96] - q[97] - q[101] + q[104] -
               q[107] + q[108] + q[114] + q[115] + q[117] + q[123] - q[125] -
               q[126] - q[127] - q[131] - q[132] - q[133] + q[134] + q[145] -
               q[147] - q[148] - q[149] - q[155] - q[156] + q[164] + q[166] -
               q[169] - q[170] + q[172] - q[179] - q[184] - q[185] - q[189] -
               q[190] - q[191] + q[192] - q[198] - q[199]);
          dYdt[2] =
              th.MW(2) *
              (q[0] - q[1] + q[3] - q[9] - 2.0 * q[10] + q[16] - q[19] - q[25] -
               q[28] + q[31] - q[33] + q[37] - q[42] - q[43] - q[49] - q[60] -
               q[61] - q[72] - q[78] + q[81] - q[98] - q[102] - q[108] -
               q[115] - q[116] - q[122] - q[128] - q[129] - q[134] - q[135] +
               q[138] - q[150] - q[157] - q[158] - q[159] - q[171] - q[172] -
               q[180] - q[192] - q[193] - q[194] - q[200]);
          dYdt[3] =
              th.MW(3) *
              (-q[0] + q[10] - q[11] - q[12] - q[13] - q[14] + q[17] + q[19] +
               q[20] + q[21] + q[22] - q[31] - q[37] - q[46] - q[50] - q[51] -
               q[64] - q[65] - q[74] - q[81] - q[82] + q[83] - q[100] - q[109] -
               q[124] - q[137] - q[138] - q[139] - q[152] - q[153] - q[161] -
               q[173] + q[174] + q[186] - q[202]);
          dYdt[4] =
              th.MW(4) *
              (q[0] + q[1] - q[2] - 2.0 * q[3] - q[8] + q[9] - 2.0 * q[15] +
               2.0 * q[18] + q[19] - q[20] + q[24] + q[25] - q[26] - q[27] -
               q[29] + q[32] - q[34] + q[42] - q[44] + q[50] - q[52] - q[53] +
               q[54] - q[62] + q[64] + q[72] - q[73] - q[79] - q[80] + q[82] +
               q[84] + q[96] + q[98] - q[99] + q[102] - q[103] + q[109] -
               q[117] - q[118] - q[123] + q[128] - q[130] - q[136] + q[140] +
               q[150] - q[151] + q[153] + q[157] - q[160] + q[162] + q[176] +
               q[180] - q[181] + q[187] + q[194] - q[195] - q[201] + q[203]);
          dYdt[5] =
              th.MW(5) * (q[2] + q[3] + q[8] + q[16] + q[20] + q[24] + q[26] +
                          q[27] - q[36] + q[44] + q[53] + q[65] + q[73] +
                          q[79] + q[80] + q[97] + q[99] + q[103] + q[130] +
                          q[136] + q[151] + q[160] + q[181] + q[195] + q[201]);
          dYdt[6] =
              th.MW(6) *
              (q[11] + q[12] + q[13] + q[14] - q[16] - q[17] - q[18] - q[19] -
               q[20] - 2.0 * q[21] - 2.0 * q[22] + q[23] + q[25] + q[26] +
               q[27] - q[32] + q[46] - q[54] + q[74] - q[75] - q[83] - q[84] +
               q[85] + q[100] + q[137] - q[140] + q[141] + q[152] + q[161] -
               q[162] + q[173] - q[174] - q[175] - q[176] + q[177] - q[186] -
               q[187] - q[196] + q[202] - q[203]);
          dYdt[7] = th.MW(7) *
                    (q[15] + q[21] + q[22] - q[23] - q[24] - q[25] - q[26] -
                     q[27] + q[75] - q[85] - q[141] + q[175] - q[177] + q[196]);
          dYdt[8] = th.MW(8) *
                    (-q[33] - q[34] - q[35] - q[36] - q[37] - q[38] - q[39] +
                     q[53] - q[56] + q[59] - q[76] - q[86] - q[104] - q[110]);
          dYdt[9] = th.MW(9) *
                    (q[35] - q[47] - q[48] - q[49] - q[50] - q[51] - q[52] -
                     q[53] - q[54] - q[55] - q[56] - 2.0 * q[57] + q[58] +
                     q[66] + q[67] + q[68] + q[79] - q[90] - q[105] - q[111] +
                     q[116] + q[122] + q[124] + q[129] + q[159] - q[164]);
          dYdt[10] = th.MW(10) *
                     (-q[58] - q[59] - q[60] - q[61] - q[62] - q[63] - q[64] -
                      q[65] - q[66] - q[67] - q[68] - q[69] + q[80] - q[91] +
                      q[97] - q[106] + q[107] - q[165] - q[166] - q[182]);
          dYdt[11] =
              th.MW(11) *
              (q[47] + q[48] + q[63] - q[77] - q[78] - q[79] - q[80] - q[81] -
               q[82] - q[83] - q[84] - q[85] - q[86] - q[87] - q[88] - q[89] -
               q[90] - q[91] - 2.0 * q[92] - 2.0 * q[93] - q[94] + q[96] +
               q[101] + q[102] + q[103] + 2.0 * q[105] + 2.0 * q[106] + q[118] -
               q[120] + q[127] + q[135] - q[143] - q[144] - q[145] + q[146] +
               q[148] + q[158] - q[167] - q[168] + q[171] + q[176] + q[182] -
               q[183] + q[190] + q[192] - q[197] + q[198] - q[204] + q[205]);
          dYdt[12] =
              th.MW(12) * (q[77] + q[83] + q[85] + q[87] + q[89] - q[101] -
                           q[102] - q[103] - q[104] - q[105] - q[106] + q[143] +
                           q[165] + q[167] + q[183] + q[185] + q[197] + q[204]);
          dYdt[13] =
              th.MW(13) *
              (-q[28] - q[29] - q[30] - q[31] - q[32] + q[33] - q[38] + q[39] +
               q[41] + q[42] + q[44] + q[45] + q[46] - q[55] + q[60] + q[64] +
               q[65] + q[69] + q[87] + q[94] + q[107] + 2.0 * q[108] +
               2.0 * q[109] + q[110] + q[111] + 2.0 * q[112] + q[116] + q[118] +
               q[119] + q[122] + q[127] + q[135] + q[142] + q[146] + q[153] +
               q[163] + q[178] + q[188]);
          dYdt[14] = th.MW(14) * (q[28] + q[29] + q[31] + q[32] - q[39] +
                                  q[43] + q[51] - q[69] + q[124] + q[129]);
          dYdt[15] =
              th.MW(15) *
              (q[34] + q[37] + q[39] - q[40] - q[41] - q[42] - q[43] - q[44] -
               q[45] - q[46] + q[49] + q[50] + q[61] + q[71] + q[72] + q[73] +
               q[74] + q[75] - q[87] - q[88] + q[89] - q[119] + q[139] -
               q[142] + q[148] + q[158] - q[163] - q[178] - q[188] + q[193]);
          dYdt[16] =
              th.MW(16) *
              (q[30] + q[36] + q[40] + q[52] + q[54] + q[62] + q[69] - q[70] -
               q[71] - q[72] - q[73] - q[74] - q[75] - q[76] + q[78] + q[82] -
               q[89] + q[95] + q[98] + q[99] + q[100] + q[139] + q[153] +
               q[159] + q[171] + q[176] + q[187] + q[200] + q[203]);
          dYdt[17] = th.MW(17) * (q[70] + q[81] + q[84] - q[95] - q[96] -
                                  q[97] - q[98] - q[99] - q[100]);
          dYdt[18] =
              th.MW(18) * (q[56] + q[57] + q[110] + q[112] - q[113] + q[114] -
                           q[115] - q[116] - q[117] - q[118] - q[119] - q[120] +
                           q[121] + q[132] + q[136] + q[137] + q[143]);
          dYdt[19] = th.MW(19) * (q[113] - q[121] - q[122] - q[123] - q[124] +
                                  q[133] + q[154] + q[165] + q[185]);
          dYdt[20] =
              th.MW(20) *
              (q[86] + q[111] - q[114] + q[119] - q[131] - q[132] - q[133] -
               q[134] - q[135] - q[136] - q[137] - q[138] - q[139] - q[140] -
               q[141] - q[142] - q[143] - q[144] - q[145] + q[156] + q[157] +
               q[160] + q[161] + q[167] + q[187] - q[205]);
          dYdt[21] =
              th.MW(21) *
              (q[90] + q[91] + q[94] + q[104] + q[131] + q[141] + q[142] -
               q[154] - q[155] - q[156] - q[157] - q[158] - q[159] - q[160] -
               q[161] - q[162] - q[163] - q[164] - q[165] - q[166] - q[167] -
               q[168] + q[170] + q[173] + q[175] + q[190]);
          dYdt[22] =
              th.MW(22) * (q[93] + q[155] + q[163] - q[169] - q[170] - q[171] -
                           q[172] - q[173] - q[174] - q[175] - q[176] - q[177] -
                           q[178] + q[179] + q[180] + q[181] + q[182] + q[183] +
                           q[193] + q[198] + q[200] + q[203] - q[205]);
          dYdt[23] = th.MW(23) * (q[92] + q[169] + q[174] + q[177] + q[178] -
                                  q[179] - q[180] - q[181] - q[182] - q[183]);
          dYdt[24] = th.MW(24) * (q[38] - q[94] - q[107] - q[108] - q[109] -
                                  q[110] - q[111] - 2.0 * q[112] + q[115] +
                                  q[126] + q[128] + q[130]);
          dYdt[25] =
              th.MW(25) * (q[55] + q[76] + q[117] + q[123] - q[125] - q[126] -
                           q[127] - q[128] - q[129] - q[130] + q[134] + q[149] +
                           q[150] + q[151] + q[152] + q[192]);
          dYdt[26] =
              th.MW(26) * (q[125] + q[138] + q[140] - q[146] - q[147] - q[148] -
                           q[149] - q[150] - q[151] - q[152] - q[153]);
          dYdt[27] = th.MW(27) * (q[88] + q[147] + q[162] + q[172]);
          dYdt[28] = th.MW(28) * (q[120] + q[145] + q[164] + q[166] - q[184] -
                                  q[185] - q[186] - q[187] - q[188] + q[191] +
                                  q[194] + q[195] + q[196] + q[197] + q[205]);
          dYdt[29] =
              th.MW(29) * (q[144] + q[184] + q[186] + q[188] - q[189] - q[190] -
                           q[191] - q[192] - q[193] - q[194] - q[195] - q[196] -
                           q[197] + q[199] + q[201] + q[202] + q[204]);
          dYdt[30] = th.MW(30) * (q[168] + q[189] - q[198] - q[199] - q[200] -
                                  q[201] - q[202] - q[203] - q[204]);
          dYdt[31] = th.MW(31) * (0.0);

          dTdt = 0.0;
          for (int n = 0; n < 32; n++) {
            dTdt -= hi[n] * dYdt[n];
            Y[n] += dYdt[n] / rho * tSub;
          }
          dTdt /= cp * rho;
          T += dTdt * tSub;

        } // End of chem sub step for loop

        // Compute d(rhoYi)/dt based on where we end up
        // Add source terms to RHS
        for (int n = 0; n < 31; n++) {
          b.dQ(i, j, k, 5 + n) += (Y[n] * rho - b.Q(i, j, k, 5 + n)) / dt;
        }

        // Store dTdt and dYdt (for implicit chem integration)
        for (int n = 0; n < 31; n++) {
          b.omega(i, j, k, n + 1) = dYdt[n] / b.Q(i, j, k, 0);
        }
        b.omega(i, j, k, 0) = dTdt;
      });
}
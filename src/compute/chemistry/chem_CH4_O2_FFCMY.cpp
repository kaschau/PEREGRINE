// ****************************************************************************
//
//  A skeletal model for methane oxygen combustion in rocket engine.
//
//  Rui Xu, Hai Wang
//  Stanford University
//
//  October 31, 2018
//
//  Please contact Hai Wang at haiwang@stanford.edu for questions and comments.
//
// ****************************************************************************
// ========================================================== //
// Y(  0) = H2
// Y(  1) = H
// Y(  2) = O2
// Y(  3) = O
// Y(  4) = OH
// Y(  5) = HO2
// Y(  6) = H2O
// Y(  7) = CH3
// Y(  8) = CH4
// Y(  9) = CO
// Y( 10) = CH2O
// Y( 11) = CO2

// 38 reactions.
// ========================================================== //

#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

void chem_CH4_O2_FFCMY(block_ &b, const thtrdat_ &th, const int &rface /*=0*/,
                       const int &indxI /*=0*/, const int &indxJ /*=0*/,
                       const int &indxK /*=0*/, const int &nChemSubSteps /*=1*/,
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
        double Y[12];
        double dYdt[12];
        double dTdt = 0.0;
        double tSub = dt / nChemSubSteps;

        // Set the initial values of Y array
        for (int n = 0; n < 11; n++) {
          Y[n] = b.q(i, j, k, 5 + n);
        }

        for (int nSub = 0; nSub < nChemSubSteps; nSub++) {

          // Compute nth species Y
          Y[11] = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < 11; n++) {
            Y[n] = fmax(fmin(Y[n], 1.0), 0.0);
            Y[11] -= Y[n];
            testSum += Y[n];
          }
          if (testSum > 1.0) {
            Y[11] = 0.0;
            for (int n = 0; n < 11; n++) {
              Y[n] /= testSum;
            }
          }

          // Concentrations
          double cs[12];
          for (int n = 0; n <= 11; n++) {
            cs[n] = rho * Y[n] / th.MW(n);
          }

          // ----------------------------------------------------------- >
          // Gibbs energy. --------------------------------------------- >
          // ----------------------------------------------------------- >

          double hi[12];
          double gbs[12];
          double cp = 0.0;
          // start scope of precomputed T**
          {
            double logT = log(T);
            double Tinv = 1.0 / T;
            double To2 = T / 2.0;
            double T2 = pow(T, 2);
            double T3 = pow(T, 3);
            double T4 = pow(T, 4);
            double T2o2 = T2 / 2.0;
            double T3o3 = T3 / 3.0;
            double T4o4 = T4 / 4.0;
            double T2o3 = T2 / 3.0;
            double T3o4 = T3 / 4.0;
            double T4o5 = T4 / 5.0;

            for (int n = 0; n <= 11; n++) {
              int m = (T <= th.NASA7(n, 0)) ? 8 : 1;
              double cps = (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T +
                            th.NASA7(n, m + 2) * T2 + th.NASA7(n, m + 3) * T3 +
                            th.NASA7(n, m + 4) * T4) *
                           th.Ru / th.MW(n);

              hi[n] = th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * To2 +
                      th.NASA7(n, m + 2) * T2o3 + th.NASA7(n, m + 3) * T3o4 +
                      th.NASA7(n, m + 4) * T4o5 + th.NASA7(n, m + 5) * Tinv;

              double scs = th.NASA7(n, m + 0) * logT + th.NASA7(n, m + 1) * T +
                           th.NASA7(n, m + 2) * T2o2 +
                           th.NASA7(n, m + 3) * T3o3 +
                           th.NASA7(n, m + 4) * T4o4 + th.NASA7(n, m + 6);

              cp += cps * Y[n];
              gbs[n] = hi[n] - scs;
            }
            // ends scope of precomputed T**
          }
          // ----------------------------------------------------------- >
          // Rate Constants. ------------------------------------------- >
          // FallOff Modifications. ------------------------------------ >
          // Forward, backward, net rates of progress. ----------------- >
          // ----------------------------------------------------------- >

          double q[38];

          // start scope of these temp vars
          {
            double cTBC, k_f, dG, K_c;

            double Fcent;
            double pmod;
            double Pr, k0;
            double A, f1, F_pdr;
            double C, N;
            double q_f, q_b;
            double logT = log(T);
            double Tinv = 1.0 / T;
            double prefRuT = 101325.0 / (th.Ru * T);
            // Reaction #0
            k_f = exp(log(109000000000.00002) - (7704.291057866103 * Tinv));
            dG = -gbs[1] - gbs[2] + gbs[3] + gbs[4];
            K_c = exp(-dG);
            q_f = k_f * cs[1] * cs[2];
            q_b = -k_f / K_c * cs[3] * cs[4];
            q[0] = q_f + q_b;

            // Reaction #1
            k_f = exp(log(3820000000.0000005) - (4000.5952913151878 * Tinv));
            dG = -gbs[0] + gbs[1] - gbs[3] + gbs[4];
            K_c = exp(-dG);
            q_f = k_f * cs[0] * cs[3];
            q_b = -k_f / K_c * cs[1] * cs[4];
            q[1] = q_f + q_b;

            // Reaction #2
            k_f = exp(log(879000000000.0001) - (9651.750652506327 * Tinv));
            dG = -gbs[0] + gbs[1] - gbs[3] + gbs[4];
            K_c = exp(-dG);
            q_f = k_f * cs[0] * cs[3];
            q_b = -k_f / K_c * cs[1] * cs[4];
            q[2] = q_f + q_b;

            // Reaction #3
            k_f = exp(log(216000.00000000003) + 1.51 * logT -
                      (1729.5655366352578 * Tinv));
            dG = -gbs[0] + gbs[1] - gbs[4] + gbs[6];
            K_c = exp(-dG);
            q_f = k_f * cs[0] * cs[4];
            q_b = -k_f / K_c * cs[1] * cs[6];
            q[3] = q_f + q_b;

            // Reaction #4
            k_f = exp(log(33.50000000000001) + 2.42 * logT -
                      (-970.2072605856204 * Tinv));
            dG = gbs[3] - 2.0 * gbs[4] + gbs[6];
            K_c = exp(-dG);
            q_f = k_f * pow(cs[4], 2.0);
            q_b = -k_f / K_c * cs[3] * cs[6];
            q[4] = q_f + q_b;

            // Reaction #5
            k_f = exp(log(4.580000000000001e+16) - 1.4 * logT -
                      (52531.08710193616 * Tinv));
            dG = -gbs[0] + 2.0 * gbs[1];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #5
            cTBC = 2.5 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] +
                   12.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.9 * cs[9] +
                   2.5 * cs[10] + 3.8 * cs[11];
            k_f *= cTBC;
            q_f = k_f * cs[0];
            q_b = -k_f / K_c * pow(cs[1], 2.0);
            q[5] = q_f + q_b;

            // Reaction #6
            k_f = exp(log(4710000000000.001) - 1.0 * logT);
            dG = -gbs[1] - gbs[3] + gbs[4];
            K_c = exp(-dG) / prefRuT;
            //  Three Body Reaction #6
            cTBC = 2.5 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] +
                   12.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.9 * cs[9] +
                   2.5 * cs[10] + 3.8 * cs[11];
            k_f *= cTBC;
            q_f = k_f * cs[1] * cs[3];
            q_b = -k_f / K_c * cs[4];
            q[6] = q_f + q_b;

            // Reaction #7
            k_f = exp(log(6.060000000000001e+24) - 3.322 * logT -
                      (60788.9196466509 * Tinv));
            dG = gbs[1] + gbs[4] - gbs[6];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #7
            cTBC = 3.0 * cs[0] + cs[1] + 1.5 * cs[2] + cs[3] + cs[4] + cs[5] +
                   cs[7] + 7.0 * cs[8] + 1.9 * cs[9] + 2.5 * cs[10] +
                   3.8 * cs[11];
            k_f *= cTBC;
            q_f = k_f * cs[6];
            q_b = -k_f / K_c * cs[1] * cs[4];
            q[7] = q_f + q_b;

            // Reaction #8
            k_f = exp(log(1.0100000000000001e+23) - 2.44 * logT -
                      (60486.98792655164 * Tinv));
            dG = gbs[1] + gbs[4] - gbs[6];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #8
            cTBC = cs[6];
            k_f *= cTBC;
            q_f = k_f * cs[6];
            q_b = -k_f / K_c * cs[1] * cs[4];
            q[8] = q_f + q_b;

            // Reaction #9
            k_f = exp(log(4650000000.000001) + 0.44 * logT);
            dG = -gbs[1] - gbs[2] + gbs[5];
            K_c = exp(-dG) / prefRuT;
            //  Three Body Reaction #9
            cTBC = 2.0 * cs[0] + cs[1] + 0.78 * cs[2] + cs[3] + cs[4] + cs[5] +
                   14.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.9 * cs[9] +
                   2.5 * cs[10] + 3.8 * cs[11];
            //  Troe Reaction #9
            Fcent = (1.0 - (0.5)) * exp(-T / (30.0)) +
                    (0.5) * exp(-T / (90000.0)) + exp(-(90000.0) * Tinv);
            C = -0.4 - 0.67 * log10(Fcent);
            N = 0.75 - 1.27 * log10(Fcent);
            k0 = exp(log(1910000000000000.2) - 1.72 * logT -
                     (264.190255086852 * Tinv));
            Pr = cTBC * k0 / k_f;
            A = log10(Pr) + C;
            f1 = A / (N - 0.14 * A);
            F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
            pmod = Pr / (1.0 + Pr) * F_pdr;
            k_f *= pmod;
            q_f = k_f * cs[1] * cs[2];
            q_b = -k_f / K_c * cs[5];
            q[9] = q_f + q_b;

            // Reaction #10
            k_f = exp(log(3680.0000000000005) + 2.087 * logT -
                      (-732.1844212407042 * Tinv));
            dG = gbs[0] - gbs[1] + gbs[2] - gbs[5];
            K_c = exp(-dG);
            q_f = k_f * cs[1] * cs[5];
            q_b = -k_f / K_c * cs[0] * cs[2];
            q[10] = q_f + q_b;

            // Reaction #11
            k_f = exp(log(70800000000.00002) - (150.96586004962973 * Tinv));
            dG = -gbs[1] + 2.0 * gbs[4] - gbs[5];
            K_c = exp(-dG);
            q_f = k_f * cs[1] * cs[5];
            q_b = -k_f / K_c * pow(cs[4], 2.0);
            q[11] = q_f + q_b;

            // Reaction #12
            k_f = 1450000000.0000002;
            dG = -gbs[1] + gbs[3] - gbs[5] + gbs[6];
            K_c = exp(-dG);
            q_f = k_f * cs[1] * cs[5];
            q_b = -k_f / K_c * cs[3] * cs[6];
            q[12] = q_f + q_b;

            // Reaction #13
            k_f = exp(log(16300000000.000002) - (-223.93269240695076 * Tinv));
            dG = gbs[2] - gbs[3] + gbs[4] - gbs[5];
            K_c = exp(-dG);
            q_f = k_f * cs[3] * cs[5];
            q_b = -k_f / K_c * cs[2] * cs[4];
            q[13] = q_f + q_b;

            // Reaction #14
            k_f = exp(log(7000000000.000001) - (-550.0189501141509 * Tinv));
            dG = gbs[2] - gbs[4] - gbs[5] + gbs[6];
            K_c = exp(-dG);
            q_f = k_f * cs[4] * cs[5];
            q_b = -k_f / K_c * cs[2] * cs[6];
            q[14] = q_f + q_b;

            // Reaction #15
            k_f = exp(log(450000000000.00006) - (5500.18950114151 * Tinv));
            dG = gbs[2] - gbs[4] - gbs[5] + gbs[6];
            K_c = exp(-dG);
            q_f = k_f * cs[4] * cs[5];
            q_b = -k_f / K_c * cs[2] * cs[6];
            q[15] = q_f + q_b;

            // Reaction #16
            k_f = exp(log(10600000000.000002) - 0.308 * logT -
                      (3493.8532210819303 * Tinv));
            dG = -gbs[3] - gbs[9] + gbs[11];
            K_c = exp(-dG) / prefRuT;
            //  Three Body Reaction #16
            cTBC = 2.5 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] +
                   12.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.9 * cs[9] +
                   2.5 * cs[10] + 3.8 * cs[11];
            //  Lindeman Reaction #16
            Fcent = 1.0;
            k0 = exp(log(1400000000000000.2) - 2.1 * logT -
                     (2767.7074342432115 * Tinv));
            Pr = cTBC * k0 / k_f;
            pmod = Pr / (1.0 + Pr);
            k_f *= pmod;
            q_f = k_f * cs[3] * cs[9];
            q_b = -k_f / K_c * cs[11];
            q[16] = q_f + q_b;

            // Reaction #17
            k_f = exp(log(2530000000.0000005) - (24003.571747891125 * Tinv));
            dG = -gbs[2] + gbs[3] - gbs[9] + gbs[11];
            K_c = exp(-dG);
            q_f = k_f * cs[2] * cs[9];
            q_b = -k_f / K_c * cs[3] * cs[11];
            q[17] = q_f + q_b;

            // Reaction #18
            k_f = exp(log(84.60000000000001) + 2.053 * logT -
                      (-179.1461539255606 * Tinv));
            dG = gbs[1] - gbs[4] - gbs[9] + gbs[11];
            K_c = exp(-dG);
            q_f = k_f * cs[4] * cs[9];
            q_b = -k_f / K_c * cs[1] * cs[11];
            q[18] = q_f + q_b;

            // Reaction #19
            k_f = exp(log(8640000000.000002) - 0.664 * logT -
                      (167.06888512159023 * Tinv));
            dG = gbs[1] - gbs[4] - gbs[9] + gbs[11];
            K_c = exp(-dG);
            q_f = k_f * cs[4] * cs[9];
            q_b = -k_f / K_c * cs[1] * cs[11];
            q[19] = q_f + q_b;

            // Reaction #20
            k_f = exp(log(157.00000000000003) + 2.18 * logT -
                      (9029.771309101852 * Tinv));
            dG = gbs[4] - gbs[5] - gbs[9] + gbs[11];
            K_c = exp(-dG);
            q_f = k_f * cs[5] * cs[9];
            q_b = -k_f / K_c * cs[4] * cs[11];
            q[20] = q_f + q_b;

            // Reaction #21
            k_f = exp(log(3070.000000000001) + 2.5 * logT -
                      (3818.4298201886345 * Tinv));
            dG = gbs[0] - gbs[1] + gbs[7] - gbs[8];
            K_c = exp(-dG);
            q_f = k_f * cs[1] * cs[8];
            q_b = -k_f / K_c * cs[0] * cs[7];
            q[21] = q_f + q_b;

            // Reaction #22
            k_f = exp(log(231000.00000000003) + 1.56 * logT -
                      (4269.817741737027 * Tinv));
            dG = -gbs[3] + gbs[4] + gbs[7] - gbs[8];
            K_c = exp(-dG);
            q_f = k_f * cs[3] * cs[8];
            q_b = -k_f / K_c * cs[4] * cs[7];
            q[22] = q_f + q_b;

            // Reaction #23
            k_f = exp(log(1000.0000000000002) + 2.182 * logT -
                      (1230.874978937981 * Tinv));
            dG = -gbs[4] + gbs[6] + gbs[7] - gbs[8];
            K_c = exp(-dG);
            q_f = k_f * cs[4] * cs[8];
            q_b = -k_f / K_c * cs[6] * cs[7];
            q[23] = q_f + q_b;

            // Reaction #24
            k_f = 141000000000.00003;
            dG = -gbs[1] - gbs[7] + gbs[8];
            K_c = exp(-dG) / prefRuT;
            //  Three Body Reaction #24
            cTBC = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] +
                   6.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.5 * cs[9] +
                   2.5 * cs[10] + 2.0 * cs[11];
            //  Troe Reaction #24
            Fcent = (1.0 - (0.37)) * exp(-T / (3315.0)) +
                    (0.37) * exp(-T / (61.0)) + exp(-(90000.0) * Tinv);
            C = -0.4 - 0.67 * log10(Fcent);
            N = 0.75 - 1.27 * log10(Fcent);
            k0 = exp(log(6.35e+29) - 5.57 * logT - (1921.2921788982876 * Tinv));
            Pr = cTBC * k0 / k_f;
            A = log10(Pr) + C;
            f1 = A / (N - 0.14 * A);
            F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
            pmod = Pr / (1.0 + Pr) * F_pdr;
            k_f *= pmod;
            q_f = k_f * cs[1] * cs[7];
            q_b = -k_f / K_c * cs[8];
            q[24] = q_f + q_b;

            // Reaction #25
            k_f = 108000000000.00002;
            dG = gbs[1] - gbs[3] - gbs[7] + gbs[10];
            K_c = exp(-dG);
            q_f = k_f * cs[3] * cs[7];
            q_b = -k_f / K_c * cs[1] * cs[10];
            q[25] = q_f + q_b;

            // Reaction #26
            k_f = 23100000000.000004;
            dG = gbs[0] + gbs[1] - gbs[3] - gbs[7] + gbs[9];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[3] * cs[7];
            q_b = -k_f / K_c * cs[0] * cs[1] * cs[9];
            q[26] = q_f;

            // Reaction #27
            k_f = exp(log(116.00000000000001) + 2.35 * logT -
                      (-765.9001299851215 * Tinv));
            dG = gbs[2] - gbs[5] - gbs[7] + gbs[8];
            K_c = exp(-dG);
            q_f = k_f * cs[5] * cs[7];
            q_b = -k_f / K_c * cs[2] * cs[8];
            q[27] = q_f + q_b;

            // Reaction #28
            k_f = exp(log(20800000000.000004) - (-296.8995247642718 * Tinv));
            dG = gbs[1] + gbs[4] - gbs[5] - gbs[7] + gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[5] * cs[7];
            q_b = -k_f / K_c * cs[1] * cs[4] * cs[10];
            q[28] = q_f;

            // Reaction #29
            k_f = exp(log(2510000000.0000005) - (14239.603139414574 * Tinv));
            dG = gbs[1] - gbs[2] + gbs[3] - gbs[7] + gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[2] * cs[7];
            q_b = -k_f / K_c * cs[1] * cs[3] * cs[10];
            q[29] = q_f;

            // Reaction #30
            k_f = exp(log(0.022800000000000004) + 2.53 * logT -
                      (4915.4484032159435 * Tinv));
            dG = -gbs[2] + gbs[4] - gbs[7] + gbs[10];
            K_c = exp(-dG);
            q_f = k_f * cs[2] * cs[7];
            q_b = -k_f / K_c * cs[4] * cs[10];
            q[30] = q_f + q_b;

            // Reaction #31
            k_f = exp(log(0.010600000000000002) + 3.36 * logT -
                      (2168.8761893796805 * Tinv));
            dG = gbs[1] - gbs[7] + gbs[8] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[7] * cs[10];
            q_b = -k_f / K_c * cs[1] * cs[8] * cs[9];
            q[31] = q_f;

            // Reaction #32
            k_f = exp(log(37000000000000.0) - (36219.729143107164 * Tinv));
            dG = gbs[0] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #32
            cTBC = 2.0 * cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] +
                   6.0 * cs[6] + cs[7] + 2.0 * cs[8] + 1.5 * cs[9] +
                   2.5 * cs[10] + 2.0 * cs[11];
            //  Troe Reaction #32
            Fcent = (1.0 - (0.932)) * exp(-T / (197.00000000000003)) +
                    (0.932) * exp(-T / (1540.0)) + exp(-(10300.0) * Tinv);
            C = -0.4 - 0.67 * log10(Fcent);
            N = 0.75 - 1.27 * log10(Fcent);
            k0 = exp(log(4.4000000000000005e+35) - 6.1 * logT -
                     (47302.636148883976 * Tinv));
            Pr = cTBC * k0 / k_f;
            A = log10(Pr) + C;
            f1 = A / (N - 0.14 * A);
            F_pdr = pow(10.0, log10(Fcent) / (1.0 + f1 * f1));
            pmod = Pr / (1.0 + Pr) * F_pdr;
            k_f *= pmod;
            q_f = k_f * cs[10];
            q_b = -k_f / K_c * cs[0] * cs[9];
            q[32] = q_f + q_b;

            // Reaction #33
            k_f = exp(log(5670000000.000001) + 0.361 * logT -
                      (2319.338829895811 * Tinv));
            dG = gbs[0] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #33
            cTBC = cs[1];
            k_f *= cTBC;
            q_f = k_f * cs[10];
            q_b = -k_f / K_c * cs[0] * cs[9];
            q[33] = q_f;

            // Reaction #34
            k_f = exp(log(11400000000.000002) + 0.582 * logT -
                      (7243.845184714733 * Tinv));
            dG = gbs[0] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            //  Three Body Reaction #34
            cTBC = cs[1];
            k_f *= cTBC;
            q_f = k_f * cs[10];
            q_b = -k_f / K_c * cs[0] * cs[9];
            q[34] = q_f;

            // Reaction #35
            k_f = exp(log(416000000.00000006) + 0.57 * logT -
                      (1389.892351523591 * Tinv));
            dG = gbs[1] - gbs[3] + gbs[4] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[3] * cs[10];
            q_b = -k_f / K_c * cs[1] * cs[4] * cs[9];
            q[35] = q_f;

            // Reaction #36
            k_f = exp(log(78200.00000000001) + 1.63 * logT -
                      (-530.8966078411978 * Tinv));
            dG = gbs[1] - gbs[4] + gbs[6] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[4] * cs[10];
            q_b = -k_f / K_c * cs[1] * cs[6] * cs[9];
            q[36] = q_f;

            // Reaction #37
            k_f = exp(log(244.00000000000006) + 2.5 * logT -
                      (18347.384191365 * Tinv));
            dG = gbs[1] - gbs[2] + gbs[5] + gbs[9] - gbs[10];
            K_c = prefRuT * exp(-dG);
            q_f = k_f * cs[2] * cs[10];
            q_b = -k_f / K_c * cs[1] * cs[5] * cs[9];
            q[37] = q_f;
          }
          // ends scope of logT and prefRuT
          // ----------------------------------------------------------- >
          // Source terms. --------------------------------------------- >
          // ----------------------------------------------------------- >

          dYdt[0] = th.MW(0) * (-q[1] - q[2] - q[3] - q[5] + q[10] + q[21] +
                                q[26] + q[32] + q[33] + q[34]);
          dYdt[1] = th.MW(1) * (-q[0] + q[1] + q[2] + q[3] + 2.0 * q[5] - q[6] +
                                q[7] + q[8] - q[9] - q[10] - q[11] - q[12] +
                                q[18] + q[19] - q[21] - q[24] + q[25] + q[26] +
                                q[28] + q[29] + q[31] + q[35] + q[36] + q[37]);
          dYdt[2] = th.MW(2) * (-q[0] - q[9] + q[10] + q[13] + q[14] + q[15] -
                                q[17] + q[27] - q[29] - q[30] - q[37]);
          dYdt[3] = th.MW(3) *
                    (q[0] - q[1] - q[2] + q[4] - q[6] + q[12] - q[13] - q[16] +
                     q[17] - q[22] - q[25] - q[26] + q[29] - q[35]);
          dYdt[4] = th.MW(4) * (q[0] + q[1] + q[2] - q[3] - 2.0 * q[4] + q[6] +
                                q[7] + q[8] + 2.0 * q[11] + q[13] - q[14] -
                                q[15] - q[18] - q[19] + q[20] + q[22] - q[23] +
                                q[28] + q[30] + q[35] - q[36]);
          dYdt[5] = th.MW(5) * (q[9] - q[10] - q[11] - q[12] - q[13] - q[14] -
                                q[15] - q[20] - q[27] - q[28] + q[37]);
          dYdt[6] = th.MW(6) * (q[3] + q[4] - q[7] - q[8] + q[12] + q[14] +
                                q[15] + q[23] + q[36]);
          dYdt[7] = th.MW(7) * (q[21] + q[22] + q[23] - q[24] - q[25] - q[26] -
                                q[27] - q[28] - q[29] - q[30] - q[31]);
          dYdt[8] = th.MW(8) * (-q[21] - q[22] - q[23] + q[24] + q[27] + q[31]);
          dYdt[9] = th.MW(9) *
                    (-q[16] - q[17] - q[18] - q[19] - q[20] + q[26] + q[31] +
                     q[32] + q[33] + q[34] + q[35] + q[36] + q[37]);
          dYdt[10] =
              th.MW(10) * (q[25] + q[28] + q[29] + q[30] - q[31] - q[32] -
                           q[33] - q[34] - q[35] - q[36] - q[37]);
          dYdt[11] = th.MW(11) * (q[16] + q[17] + q[18] + q[19] + q[20]);

          dTdt = 0.0;
          for (int n = 0; n < 12; n++) {
            dTdt -= hi[n] * dYdt[n];
            Y[n] += dYdt[n] / rho * tSub;
          }
          dTdt /= cp * rho;
          T += dTdt * tSub;

        } // End of chem sub step for loop

        // Compute d(rhoYi)/dt based on where we end up
        // Add source terms to RHS
        for (int n = 0; n < 11; n++) {
          b.dQ(i, j, k, 5 + n) += (Y[n] * rho - b.Q(i, j, k, 5 + n)) / dt;
        }

        // Store dTdt and dYdt (for implicit chem integration)
        for (int n = 0; n < 11; n++) {
          b.omega(i, j, k, n + 1) = dYdt[n] / b.Q(i, j, k, 0);
        }
        b.omega(i, j, k, 0) = dTdt;
      });
}

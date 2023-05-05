// ****************************************************************************
//
//    A 22-Species Reduced Mechanism for C2H4-air
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
// Y(  8) = CH3
// Y(  9) = CH4
// Y( 10) = CO
// Y( 11) = CO2
// Y( 12) = CH2O
// Y( 13) = C2H2
// Y( 14) = C2H4
// Y( 15) = C2H6
// Y( 16) = HCCO
// Y( 17) = CH2CO
// Y( 18) = CH3CHO
// Y( 19) = aC3H5
// Y( 20) = C3H6
// Y( 21) = N2

// 206 reactions.
// ========================================================== //

#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>

void chem_C2H4_Air_Red22(block_ &b, const thtrdat_ &th, const int &rface /*=0*/,
                         const int &indxI /*=0*/, const int &indxJ /*=0*/,
                         const int &indxK /*=0*/,
                         const int &nChemSubSteps /*=1*/,
                         const double &dt /*=1.0*/) {

  // THIS DOES NOT WORK WITH CHEMICAL SUB STEPS!!!!!

  // --------------------------------------------------------------|
  // cc range
  // --------------------------------------------------------------|
  MDRange3 range = getRange3(b, rface, indxI, indxJ, indxK);

  Kokkos::parallel_for(
      "Compute chemical source terms", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double p = b.q(i, j, k, 0) * 10.0; // convert to dynes/cm^2
        double T = b.q(i, j, k, 4);
        double &rho = b.Q(i, j, k, 0);
        double Y[22];
        double dYdt[22];

        double tSub = dt / nChemSubSteps;
        double dTdt = 0.0;

        // Set the initial values of Y array
        for (int n = 0; n < 21; n++) {
          Y[n] = b.q(i, j, k, 5 + n);
        }

        for (int nSub = 0; nSub < nChemSubSteps; nSub++) {

          // Compute nth species Y
          Y[21] = 1.0;
          double testSum = 0.0;
          for (int n = 0; n < 21; n++) {
            Y[n] = fmax(fmin(Y[n], 1.0), 0.0);
            Y[21] -= Y[n];
            testSum += Y[n];
            dYdt[n] = 0.0;
          }
          dYdt[21] = 0.0;
          if (testSum > 1.0) {
            Y[21] = 0.0;
            for (int n = 0; n < 21; n++) {
              Y[n] /= testSum;
            }
          }

          // Concentrations
          double cs[22];
          cs[0] = Y[0] / 2.01593995e0;
          cs[1] = Y[1] / 1.00796998e0;
          cs[2] = Y[2] / 1.59994001e1;
          cs[3] = Y[3] / 3.19988003e1;
          cs[4] = Y[4] / 1.70073701e1;
          cs[5] = Y[5] / 1.80153401e1;
          cs[6] = Y[6] / 3.30067703e1;
          cs[7] = Y[7] / 3.40147402e1;
          cs[8] = Y[8] / 1.50350603e1;
          cs[9] = Y[9] / 1.60430303e1;
          cs[10] = Y[10] / 2.80105505e1;
          cs[11] = Y[11] / 4.40099506e1;
          cs[12] = Y[12] / 3.00264904e1;
          cs[13] = Y[13] / 2.60382407e1;
          cs[14] = Y[14] / 2.80541806e1;
          cs[15] = Y[15] / 3.00701206e1;
          cs[16] = Y[16] / 4.10296708e1;
          cs[17] = Y[17] / 4.20376408e1;
          cs[18] = Y[18] / 4.40535808e1;
          cs[19] = Y[19] / 4.10733010e1;
          cs[20] = Y[20] / 4.20812709e1;
          cs[21] = Y[21] / 2.80133991e1;

          double SUM = 0.0;
          for (int n = 0; n < 22; n++) {
            SUM += cs[n];
          }
          SUM = p / (SUM * T * 8.314510e7);
          for (int n = 0; n < 22; n++) {
            cs[n] *= SUM;
          }
          double CTOT = 0.0;
          for (int n = 0; n < 22; n++) {
            CTOT += cs[n];
          }
          double cp = 0.0;
          double hi[22];
          double logT = log(T);
          double Tinv = 1.0 / T;
          double EG[31];
          // start scope of precomputed T**
          {
            double TN1 = logT - 1.0;
            double To2 = T / 2.0;
            double T2 = T * T;
            double T3 = T2 * T;
            double T4 = T3 * T;
            double T2o3 = T2 / 3.0;
            double T3o4 = T3 / 4.0;
            double T4o5 = T4 / 5.0;

            for (int n = 0; n <= 21; n++) {
              int m = (T <= th.NASA7(n, 0)) ? 8 : 1;
              double cps = (th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * T +
                            th.NASA7(n, m + 2) * T2 + th.NASA7(n, m + 3) * T3 +
                            th.NASA7(n, m + 4) * T4) *
                           th.Ru / th.MW(n);
              hi[n] = th.NASA7(n, m + 0) + th.NASA7(n, m + 1) * To2 +
                      th.NASA7(n, m + 2) * T2o3 + th.NASA7(n, m + 3) * T3o4 +
                      th.NASA7(n, m + 4) * T4o5 + th.NASA7(n, m + 5) * Tinv;
              cp += cps * Y[n];
            }
            // ends scope of precomputed T**

            if (T > 1000.0) {
              double SMH;
              SMH = -3.20502331e0 + 9.50158922e2 * Tinv + 3.3372792e0 * TN1 -
                    2.47012366e-5 * T + 8.32427963e-8 * T2 -
                    1.49638662e-11 * T3 + 1.00127688e-15 * T4;
              EG[0] = exp(SMH);
              SMH = -4.46682914e-1 - 2.54736599e4 * Tinv + 2.50000001e0 * TN1 -
                    1.15421486e-11 * T + 2.69269913e-15 * T2 -
                    3.94596029e-19 * T3 + 2.49098679e-23 * T4;
              EG[1] = exp(SMH);
              SMH = 4.78433864e0 - 2.92175791e4 * Tinv + 2.56942078e0 * TN1 -
                    4.29870568e-5 * T + 6.99140982e-9 * T2 -
                    8.34814992e-13 * T3 + 6.14168455e-17 * T4;
              EG[2] = exp(SMH);
              SMH = 5.45323129e0 + 1.08845772e3 * Tinv + 3.28253784e0 * TN1 +
                    7.4154377e-4 * T - 1.26327778e-7 * T2 +
                    1.74558796e-11 * T3 - 1.08358897e-15 * T4;
              EG[3] = exp(SMH);
              SMH = 4.4766961e0 - 3.858657e3 * Tinv + 3.09288767e0 * TN1 +
                    2.74214858e-4 * T + 2.10842047e-8 * T2 -
                    7.3288463e-12 * T3 + 5.8706188e-16 * T4;
              EG[4] = exp(SMH);
              SMH = 4.9667701e0 + 3.00042971e4 * Tinv + 3.03399249e0 * TN1 +
                    1.08845902e-3 * T - 2.73454197e-8 * T2 -
                    8.08683225e-12 * T3 + 8.4100496e-16 * T4;
              EG[5] = exp(SMH);
              SMH = 3.78510215e0 - 1.11856713e2 * Tinv + 4.0172109e0 * TN1 +
                    1.11991007e-3 * T - 1.05609692e-7 * T2 +
                    9.52053083e-12 * T3 - 5.39542675e-16 * T4;
              EG[6] = exp(SMH);
              SMH = 2.91615662e0 + 1.78617877e4 * Tinv + 4.16500285e0 * TN1 +
                    2.45415847e-3 * T - 3.16898708e-7 * T2 +
                    3.09321655e-11 * T3 - 1.43954153e-15 * T4;
              EG[7] = exp(SMH);
              SMH = 5.48497999e0 - 7.10124364e4 * Tinv + 2.87846473e0 * TN1 +
                    4.85456841e-4 * T + 2.40742758e-8 * T2 -
                    1.08906541e-11 * T3 + 8.80396915e-16 * T4;
              EG[8] = exp(SMH);
              SMH = 6.17119324e0 - 4.6263604e4 * Tinv + 2.87410113e0 * TN1 +
                    1.82819646e-3 * T - 2.34824328e-7 * T2 +
                    2.16816291e-11 * T3 - 9.38637835e-16 * T4;
              EG[9] = exp(SMH);
              SMH = 8.62650169e0 - 5.09259997e4 * Tinv + 2.29203842e0 * TN1 +
                    2.32794319e-3 * T - 3.35319912e-7 * T2 + 3.48255e-11 * T3 -
                    1.69858183e-15 * T4;
              EG[10] = exp(SMH);
              SMH = 8.48007179e0 - 1.67755843e4 * Tinv + 2.28571772e0 * TN1 +
                    3.61995019e-3 * T - 4.97857247e-7 * T2 +
                    4.9640387e-11 * T3 - 2.33577197e-15 * T4;
              EG[11] = exp(SMH);
              SMH = 1.8437318e1 + 9.46834459e3 * Tinv + 7.4851495e-2 * TN1 +
                    6.69547335e-3 * T - 9.55476348e-7 * T2 +
                    1.01910446e-10 * T3 - 5.0907615e-15 * T4;
              EG[12] = exp(SMH);
              SMH = 7.81868772e0 + 1.41518724e4 * Tinv + 2.71518561e0 * TN1 +
                    1.03126372e-3 * T - 1.66470962e-7 * T2 +
                    1.9171084e-11 * T3 - 1.01823858e-15 * T4;
              EG[13] = exp(SMH);
              SMH = 2.27163806e0 + 4.8759166e4 * Tinv + 3.85746029e0 * TN1 +
                    2.20718513e-3 * T - 3.69135673e-7 * T2 +
                    4.36241823e-11 * T3 - 2.36042082e-15 * T4;
              EG[14] = exp(SMH);
              SMH = 9.79834492e0 - 4.01191815e3 * Tinv + 2.77217438e0 * TN1 +
                    2.47847763e-3 * T - 4.14076022e-7 * T2 +
                    4.90968148e-11 * T3 - 2.66754356e-15 * T4;
              EG[15] = exp(SMH);
              SMH = 1.3656323e1 + 1.39958323e4 * Tinv + 1.76069008e0 * TN1 +
                    4.60000041e-3 * T - 7.37098022e-7 * T2 +
                    8.38676767e-11 * T3 - 4.4192782e-15 * T4;
              EG[16] = exp(SMH);
              SMH = 2.929575e0 - 1.2783252e2 * Tinv + 3.770799e0 * TN1 +
                    3.9357485e-3 * T - 4.42730667e-7 * T2 +
                    3.28702583e-11 * T3 - 1.056308e-15 * T4;
              EG[17] = exp(SMH);
              SMH = -1.23028121e0 - 2.59359992e4 * Tinv + 4.14756964e0 * TN1 +
                    2.98083332e-3 * T - 3.9549142e-7 * T2 +
                    3.89510143e-11 * T3 - 1.80617607e-15 * T4;
              EG[18] = exp(SMH);
              SMH = 6.4023701e-1 - 4.8316688e4 * Tinv + 4.278034e0 * TN1 +
                    2.3781402e-3 * T - 2.71683483e-7 * T2 + 2.1219005e-11 * T3 -
                    7.4431895e-16 * T4;
              EG[19] = exp(SMH);
              SMH = 7.78732378e0 - 3.46128739e4 * Tinv + 3.016724e0 * TN1 +
                    5.1651146e-3 * T - 7.80137248e-7 * T2 + 8.480274e-11 * T3 -
                    4.3130352e-15 * T4;
              EG[20] = exp(SMH);
              SMH = 1.03053693e1 - 4.93988614e3 * Tinv + 2.03611116e0 * TN1 +
                    7.32270755e-3 * T - 1.11846319e-6 * T2 +
                    1.22685769e-10 * T3 - 6.28530305e-15 * T4;
              EG[21] = exp(SMH);
              SMH = 1.34624343e1 - 1.285752e4 * Tinv + 1.95465642e0 * TN1 +
                    8.6986361e-3 * T - 1.33034445e-6 * T2 +
                    1.46014741e-10 * T3 - 7.4820788e-15 * T4;
              EG[22] = exp(SMH);
              SMH = 1.51156107e1 + 1.14263932e4 * Tinv + 1.0718815e0 * TN1 +
                    1.08426339e-2 * T - 1.67093445e-6 * T2 +
                    1.84510001e-10 * T3 - 9.5001445e-15 * T4;
              EG[23] = exp(SMH);
              SMH = -3.9302595e0 - 1.9327215e4 * Tinv + 5.6282058e0 * TN1 +
                    2.04267005e-3 * T - 2.65575783e-7 * T2 +
                    2.38550433e-11 * T3 - 9.703916e-16 * T4;
              EG[24] = exp(SMH);
              SMH = 6.32247205e-1 + 7.55105311e3 * Tinv + 4.51129732e0 * TN1 +
                    4.50179872e-3 * T - 6.94899392e-7 * T2 +
                    7.69454902e-11 * T3 - 3.974191e-15 * T4;
              EG[25] = exp(SMH);
              SMH = -5.0320879e0 - 4.9032178e2 * Tinv + 5.9756699e0 * TN1 +
                    4.0652957e-3 * T - 4.5727075e-7 * T2 + 3.39192008e-11 * T3 -
                    1.08800855e-15 * T4;
              EG[26] = exp(SMH);
              SMH = -3.4807917e0 + 2.2593122e4 * Tinv + 5.4041108e0 * TN1 +
                    5.8615295e-3 * T - 7.04385617e-7 * T2 +
                    5.69770425e-11 * T3 - 2.04924315e-15 * T4;
              EG[27] = exp(SMH);
              SMH = -1.124305e1 - 1.7482449e4 * Tinv + 6.5007877e0 * TN1 +
                    7.1623655e-3 * T - 9.46360533e-7 * T2 +
                    9.23400083e-11 * T3 - 4.51819435e-15 * T4;
              EG[28] = exp(SMH);
              SMH = -1.331335e1 + 9.235703e2 * Tinv + 6.732257e0 * TN1 +
                    7.45417e-3 * T - 8.24983167e-7 * T2 + 6.01001833e-11 * T3 -
                    1.883102e-15 * T4;
              EG[29] = exp(SMH);
              SMH = -1.5515297e1 - 7.9762236e3 * Tinv + 7.7097479e0 * TN1 +
                    8.0157425e-3 * T - 8.78670633e-7 * T2 +
                    6.32402933e-11 * T3 - 1.94313595e-15 * T4;
              EG[30] = exp(SMH);
            } else {
              double SMH;
              SMH = 6.83010238e-1 + 9.17935173e2 * Tinv + 2.34433112e0 * TN1 +
                    3.99026037e-3 * T - 3.2463585e-6 * T2 + 1.67976745e-9 * T3 -
                    3.68805881e-13 * T4;
              EG[0] = exp(SMH);
              SMH = -4.46682853e-1 - 2.54736599e4 * Tinv + 2.5e0 * TN1 +
                    3.5266641e-13 * T - 3.32653273e-16 * T2 +
                    1.91734693e-19 * T3 - 4.63866166e-23 * T4;
              EG[1] = exp(SMH);
              SMH = 2.05193346e0 - 2.91222592e4 * Tinv + 3.1682671e0 * TN1 -
                    1.63965942e-3 * T + 1.10717733e-6 * T2 -
                    5.10672187e-10 * T3 + 1.05632986e-13 * T4;
              EG[2] = exp(SMH);
              SMH = 3.65767573e0 + 1.06394356e3 * Tinv + 3.78245636e0 * TN1 -
                    1.49836708e-3 * T + 1.641217e-6 * T2 - 8.06774591e-10 * T3 +
                    1.62186419e-13 * T4;
              EG[3] = exp(SMH);
              SMH = -1.03925458e-1 - 3.61508056e3 * Tinv + 3.99201543e0 * TN1 -
                    1.20065876e-3 * T + 7.69656402e-7 * T2 -
                    3.23427778e-10 * T3 + 6.8205735e-14 * T4;
              EG[4] = exp(SMH);
              SMH = -8.49032208e-1 + 3.02937267e4 * Tinv + 4.19864056e0 * TN1 -
                    1.01821705e-3 * T + 1.08673369e-6 * T2 -
                    4.57330885e-10 * T3 + 8.85989085e-14 * T4;
              EG[5] = exp(SMH);
              SMH = 3.71666245e0 - 2.9480804e2 * Tinv + 4.30179801e0 * TN1 -
                    2.37456026e-3 * T + 3.52638152e-6 * T2 -
                    2.02303245e-9 * T3 + 4.64612562e-13 * T4;
              EG[6] = exp(SMH);
              SMH = 3.43505074e0 + 1.77025821e4 * Tinv + 4.27611269e0 * TN1 -
                    2.71411208e-4 * T + 2.78892835e-6 * T2 -
                    1.79809011e-9 * T3 + 4.31227181e-13 * T4;
              EG[7] = exp(SMH);
              SMH = 2.08401108e0 - 7.07972934e4 * Tinv + 3.48981665e0 * TN1 +
                    1.61917771e-4 * T - 2.81498442e-7 * T2 +
                    2.63514439e-10 * T3 - 7.03045335e-14 * T4;
              EG[8] = exp(SMH);
              SMH = 1.56253185e0 - 4.60040401e4 * Tinv + 3.76267867e0 * TN1 +
                    4.84436072e-4 * T + 4.65816402e-7 * T2 -
                    3.20909294e-10 * T3 + 8.43708595e-14 * T4;
              EG[9] = exp(SMH);
              SMH = -7.69118967e-1 - 5.04968163e4 * Tinv + 4.19860411e0 * TN1 -
                    1.1833071e-3 * T + 1.37216037e-6 * T2 -
                    5.57346651e-10 * T3 + 9.71573685e-14 * T4;
              EG[10] = exp(SMH);
              SMH = 1.60456433e0 - 1.64449988e4 * Tinv + 3.6735904e0 * TN1 +
                    1.00547587e-3 * T + 9.55036427e-7 * T2 -
                    5.72597854e-10 * T3 + 1.27192867e-13 * T4;
              EG[11] = exp(SMH);
              SMH = -4.64130376e0 + 1.02466476e4 * Tinv + 5.14987613e0 * TN1 -
                    6.8354894e-3 * T + 8.19667665e-6 * T2 - 4.03952522e-9 * T3 +
                    8.3346978e-13 * T4;
              EG[12] = exp(SMH);
              SMH = 3.50840928e0 + 1.4344086e4 * Tinv + 3.57953347e0 * TN1 -
                    3.0517684e-4 * T + 1.69469055e-7 * T2 +
                    7.55838237e-11 * T3 - 4.52212249e-14 * T4;
              EG[13] = exp(SMH);
              SMH = 9.90105222e0 + 4.83719697e4 * Tinv + 2.35677352e0 * TN1 +
                    4.49229838e-3 * T - 1.18726045e-6 * T2 +
                    2.04932518e-10 * T3 - 7.1849774e-15 * T4;
              EG[14] = exp(SMH);
              SMH = 3.39437243e0 - 3.83956496e3 * Tinv + 4.22118584e0 * TN1 -
                    1.62196266e-3 * T + 2.29665743e-6 * T2 -
                    1.10953411e-9 * T3 + 2.16884432e-13 * T4;
              EG[15] = exp(SMH);
              SMH = 6.028129e-1 + 1.43089567e4 * Tinv + 4.79372315e0 * TN1 -
                    4.95416684e-3 * T + 6.22033347e-6 * T2 -
                    3.16071051e-9 * T3 + 6.5886326e-13 * T4;
              EG[16] = exp(SMH);
              SMH = 1.3152177e1 - 9.786011e2 * Tinv + 2.106204e0 * TN1 +
                    3.6082975e-3 * T + 8.89745333e-7 * T2 - 6.14803e-10 * T3 +
                    1.037805e-13 * T4;
              EG[17] = exp(SMH);
              SMH = 1.39397051e1 - 2.64289807e4 * Tinv + 8.08681094e-1 * TN1 +
                    1.16807815e-2 * T - 5.91953025e-6 * T2 +
                    2.33460364e-9 * T3 - 4.25036487e-13 * T4;
              EG[18] = exp(SMH);
              SMH = 5.920391e0 - 4.8621794e4 * Tinv + 3.2815483e0 * TN1 +
                    3.48823955e-3 * T - 3.975874e-7 * T2 - 1.00870267e-10 * T3 +
                    4.90947725e-14 * T4;
              EG[19] = exp(SMH);
              SMH = 8.51054025e0 - 3.48598468e4 * Tinv + 3.21246645e0 * TN1 +
                    7.5739581e-4 * T + 4.32015687e-6 * T2 - 2.98048206e-9 * T3 +
                    7.35754365e-13 * T4;
              EG[20] = exp(SMH);
              SMH = 4.09733096e0 - 5.08977593e3 * Tinv + 3.95920148e0 * TN1 -
                    3.78526124e-3 * T + 9.51650487e-6 * T2 -
                    5.76323961e-9 * T3 + 1.34942187e-12 * T4;
              EG[21] = exp(SMH);
              SMH = 4.70720924e0 - 1.28416265e4 * Tinv + 4.30646568e0 * TN1 -
                    2.09329446e-3 * T + 8.28571345e-6 * T2 -
                    4.99272172e-9 * T3 + 1.15254502e-12 * T4;
              EG[22] = exp(SMH);
              SMH = 2.66682316e0 + 1.15222055e4 * Tinv + 4.29142492e0 * TN1 -
                    2.75077135e-3 * T + 9.99063813e-6 * T2 -
                    5.90388571e-9 * T3 + 1.34342886e-12 * T4;
              EG[23] = exp(SMH);
              SMH = 1.2490417e1 - 2.0059449e4 * Tinv + 2.2517214e0 * TN1 +
                    8.8275105e-3 * T - 3.95485017e-6 * T2 + 1.43964658e-9 * T3 -
                    2.53324055e-13 * T4;
              EG[24] = exp(SMH);
              SMH = 1.2215648e1 + 7.04291804e3 * Tinv + 2.1358363e0 * TN1 +
                    9.05943605e-3 * T - 2.89912457e-6 * T2 +
                    7.7866464e-10 * T3 - 1.00728807e-13 * T4;
              EG[25] = exp(SMH);
              SMH = 9.5714535e0 - 1.5214766e3 * Tinv + 3.4090624e0 * TN1 +
                    5.369287e-3 * T + 3.1524875e-7 * T2 + 5.96548592e-10 * T3 +
                    1.43369255e-13 * T4;
              EG[26] = exp(SMH);
              SMH = 4.1030159e0 + 2.1572878e4 * Tinv + 4.7294595e0 * TN1 -
                    1.5966429e-3 * T + 7.92248683e-6 * T2 - 4.78821758e-9 * T3 +
                    1.0965556e-12 * T4;
              EG[27] = exp(SMH);
              SMH = 1.7173214e1 - 1.9245629e4 * Tinv + 1.3631835e0 * TN1 +
                    9.9069105e-3 * T + 2.08284333e-6 * T2 - 2.77962958e-9 * T3 +
                    7.9232855e-13 * T4;
              EG[28] = exp(SMH);
              SMH = 1.614534e1 - 1.074826e3 * Tinv + 1.493307e0 * TN1 +
                    1.046259e-2 * T + 7.47799e-7 * T2 - 1.39076e-9 * T3 +
                    3.579073e-13 * T4;
              EG[29] = exp(SMH);
              SMH = 2.1136034e1 - 1.0312346e4 * Tinv + 1.0491173e0 * TN1 +
                    1.30044865e-2 * T + 3.92375267e-7 * T2 -
                    1.63292767e-9 * T3 + 4.68601035e-13 * T4;
              EG[30] = exp(SMH);
            }
          }

          double RF[206], RB[206];

          { // scope start
            double EQK;
            // HACK: IS THIS RIGHT FOR PATM?
            double prefRuT = 1013250.0 / (8.31451e7 * T);

            RF[0] = exp(3.20498617e1 - 7.25286183e3 * Tinv);
            EQK = EG[2] * EG[4] / EG[1] / EG[3];
            RB[0] = RF[0] / EQK;
            RF[1] = exp(1.08197783e1 + 2.67e0 * logT - 3.16523284e3 * Tinv);
            EQK = EG[1] * EG[4] / EG[0] / EG[2];
            RB[1] = RF[1] / EQK;
            RF[2] = exp(1.9190789e1 + 1.51e0 * logT - 1.72603317e3 * Tinv);
            EQK = EG[1] * EG[5] / EG[0] / EG[4];
            RB[2] = RF[2] / EQK;
            RF[3] = exp(1.0482906e1 + 2.4e0 * logT + 1.06178717e3 * Tinv);
            EQK = EG[2] * EG[5] / EG[4] / EG[4];
            RB[3] = RF[3] / EQK;
            RF[4] = exp(4.14465317e1 - 1e0 * logT);
            EQK = EG[0] / EG[1] / EG[1] / prefRuT;
            RB[4] = RF[4] / EQK;
            RF[5] = exp(3.90385861e1 - 6e-1 * logT);
            EQK = EG[0] / EG[1] / EG[1] / prefRuT;
            RB[5] = RF[5] / EQK;
            RF[6] = exp(4.55408762e1 - 1.25e0 * logT);
            EQK = EG[0] / EG[1] / EG[1] / prefRuT;
            RB[6] = RF[6] / EQK;
            RF[7] = exp(4.775645e1 - 2e0 * logT);
            EQK = EG[0] / EG[1] / EG[1] / prefRuT;
            RB[7] = RF[7] / EQK;
            RF[8] = 4e1 * RF[7];
            EQK = EG[5] / EG[1] / EG[4] / prefRuT;
            RB[8] = RF[8] / EQK;
            RF[9] = 5e-1 * RF[4];
            EQK = EG[4] / EG[1] / EG[2] / prefRuT;
            RB[9] = RF[9] / EQK;
            RF[10] = 1.2e-1 * RF[4];
            EQK = EG[3] / EG[2] / EG[2] / prefRuT;
            RB[10] = RF[10] / EQK;
            RF[11] = exp(4.24761511e1 - 8.6e-1 * logT);
            EQK = EG[6] / EG[1] / EG[3] / prefRuT;
            RB[11] = RF[11] / EQK;
            RF[12] = exp(4.71503141e1 - 1.72e0 * logT);
            EQK = EG[6] / EG[1] / EG[3] / prefRuT;
            RB[12] = RF[12] / EQK;
            RF[13] = exp(4.42511034e1 - 7.6e-1 * logT);
            EQK = EG[6] / EG[1] / EG[3] / prefRuT;
            RB[13] = RF[13] / EQK;
            RF[14] = exp(4.47046282e1 - 1.24e0 * logT);
            EQK = EG[6] / EG[1] / EG[3] / prefRuT;
            RB[14] = RF[14] / EQK;
            RF[15] = exp(3.19350862e1 - 3.7e-1 * logT);
            EQK = EG[7] / EG[4] / EG[4] / prefRuT;
            RB[15] = RF[15] / EQK;
            RF[16] = exp(2.90097872e1 - 3.37658384e2 * Tinv);
            EQK = EG[2] * EG[5] / EG[1] / EG[6];
            RB[16] = RF[16] / EQK;
            RF[17] = exp(3.04404238e1 - 4.12637667e2 * Tinv);
            EQK = EG[0] * EG[3] / EG[1] / EG[6];
            RB[17] = RF[17] / EQK;
            RF[18] = exp(3.18908801e1 - 1.50965e2 * Tinv);
            EQK = EG[4] * EG[4] / EG[1] / EG[6];
            RB[18] = RF[18] / EQK;
            RF[19] = 2e13;
            EQK = EG[3] * EG[4] / EG[2] / EG[6];
            RB[19] = RF[19] / EQK;
            RF[20] = exp(3.14683206e1 + 2.51608334e2 * Tinv);
            EQK = EG[3] * EG[5] / EG[4] / EG[6];
            RB[20] = RF[20] / EQK;
            RF[21] = exp(2.55908003e1 + 8.20243168e2 * Tinv);
            EQK = EG[3] * EG[7] / EG[6] / EG[6];
            RB[21] = RF[21] / EQK;
            RF[22] = exp(3.36712758e1 - 6.03860001e3 * Tinv);
            EQK = EG[3] * EG[7] / EG[6] / EG[6];
            RB[22] = RF[22] / EQK;
            RF[23] = exp(1.6308716e1 + 2e0 * logT - 2.61672667e3 * Tinv);
            EQK = EG[0] * EG[6] / EG[1] / EG[7];
            RB[23] = RF[23] / EQK;
            RF[24] = exp(2.99336062e1 - 1.81158e3 * Tinv);
            EQK = EG[4] * EG[5] / EG[1] / EG[7];
            RB[24] = RF[24] / EQK;
            RF[25] = exp(1.60803938e1 + 2e0 * logT - 2.01286667e3 * Tinv);
            EQK = EG[4] * EG[6] / EG[2] / EG[7];
            RB[25] = RF[25] / EQK;
            RF[26] = exp(2.81906369e1 - 1.61029334e2 * Tinv);
            EQK = EG[5] * EG[6] / EG[4] / EG[7];
            RB[26] = RF[26] / EQK;
            RF[27] = exp(3.39940492e1 - 4.81075134e3 * Tinv);
            EQK = EG[5] * EG[6] / EG[4] / EG[7];
            RB[27] = RF[27] / EQK;
            RF[28] = exp(3.40312786e1 - 1.50965e3 * Tinv);
            EQK = EG[14] / EG[2] / EG[13] / prefRuT;
            RB[28] = RF[28] / EQK;
            RF[29] = exp(1.76783433e1 + 1.228e0 * logT - 3.52251667e1 * Tinv);
            EQK = EG[1] * EG[14] / EG[4] / EG[13];
            RB[29] = RF[29] / EQK;
            RF[30] = exp(1.75767107e1 + 1.5e0 * logT - 4.00560467e4 * Tinv);
            EQK = EG[16] / EG[0] / EG[13] / prefRuT;
            RB[30] = RF[30] / EQK;
            RF[31] = exp(2.85473118e1 - 2.40537567e4 * Tinv);
            EQK = EG[2] * EG[14] / EG[3] / EG[13];
            RB[31] = RF[31] / EQK;
            RF[32] = exp(3.26416564e1 - 1.18759134e4 * Tinv);
            EQK = EG[4] * EG[14] / EG[6] / EG[13];
            RB[32] = RF[32] / EQK;
            RF[33] = 5.7e13;
            EQK = EG[1] * EG[13] / EG[2] / EG[8];
            RB[33] = RF[33] / EQK;
            RF[34] = 3e13;
            EQK = EG[1] * EG[15] / EG[4] / EG[8];
            RB[34] = RF[34] / EQK;
            RF[35] = exp(1.85223344e1 + 1.79e0 * logT - 8.40371835e2 * Tinv);
            EQK = EG[1] * EG[9] / EG[0] / EG[8];
            RB[35] = RF[35] / EQK;
            RF[36] = exp(2.93732401e1 + 3.79928584e2 * Tinv);
            EQK = EG[1] * EG[16] / EG[5] / EG[8];
            RB[36] = RF[36] / EQK;
            RF[37] = 3.3e13;
            EQK = EG[2] * EG[15] / EG[3] / EG[8];
            RB[37] = RF[37] / EQK;
            RF[38] = 5e13;
            EQK = EG[24] / EG[8] / EG[13] / prefRuT;
            RB[38] = RF[38] / EQK;
            RF[39] = exp(2.88547965e1 - 3.47219501e2 * Tinv);
            EQK = EG[13] * EG[15] / EG[8] / EG[14];
            RB[39] = RF[39] / EQK;
            RF[40] = exp(2.77171988e1 + 4.8e-1 * logT + 1.30836334e2 * Tinv);
            EQK = EG[16] / EG[1] / EG[15] / prefRuT;
            RB[40] = RF[40] / EQK;
            RF[41] = 7.34e13;
            EQK = EG[0] * EG[13] / EG[1] / EG[15];
            RB[41] = RF[41] / EQK;
            RF[42] = 3e13;
            EQK = EG[4] * EG[13] / EG[2] / EG[15];
            RB[42] = RF[42] / EQK;
            RF[43] = 3e13;
            EQK = EG[1] * EG[14] / EG[2] / EG[15];
            RB[43] = RF[43] / EQK;
            RF[44] = 5e13;
            EQK = EG[5] * EG[13] / EG[4] / EG[15];
            RB[44] = RF[44] / EQK;
            RF[45] = exp(3.9769885e1 - 1e0 * logT - 8.55468335e3 * Tinv);
            EQK = EG[1] * EG[13] / EG[15] * prefRuT;
            RB[45] = RF[45] / EQK;
            RF[46] = exp(2.96591694e1 - 2.01286667e2 * Tinv);
            EQK = EG[6] * EG[13] / EG[3] / EG[15];
            RB[46] = RF[46] / EQK;
            RF[47] = exp(3.77576522e1 - 8e-1 * logT);
            EQK = EG[11] / EG[1] / EG[9] / prefRuT;
            RB[47] = RF[47] / EQK;
            RF[48] = exp(1.31223634e1 + 2e0 * logT - 3.63825651e3 * Tinv);
            EQK = EG[1] * EG[11] / EG[0] / EG[9];
            RB[48] = RF[48] / EQK;
            RF[49] = 8e13;
            EQK = EG[1] * EG[15] / EG[2] / EG[9];
            RB[49] = RF[49] / EQK;
            RF[50] = exp(2.99880944e1 - 7.54825001e2 * Tinv);
            EQK = EG[4] * EG[15] / EG[3] / EG[9];
            RB[50] = RF[50] / EQK;
            RF[51] = 2.5e-1 * RF[50];
            EQK = EG[1] * EG[1] * EG[14] / EG[3] / EG[9] * prefRuT;
            RB[51] = RF[51] / EQK;
            RF[52] = 2e13;
            EQK = EG[1] * EG[16] / EG[4] / EG[9];
            RB[52] = RF[52] / EQK;
            RF[53] = exp(1.62403133e1 + 2e0 * logT - 1.50965e3 * Tinv);
            EQK = EG[5] * EG[8] / EG[4] / EG[9];
            RB[53] = RF[53] / EQK;
            RF[54] = 2e13;
            EQK = EG[4] * EG[16] / EG[6] / EG[9];
            RB[54] = RF[54] / EQK;
            RF[55] = exp(2.74203001e1 + 5e-1 * logT - 2.26950717e3 * Tinv);
            EQK = EG[25] / EG[9] / EG[13] / prefRuT;
            RB[55] = RF[55] / EQK;
            RF[56] = 4e13;
            EQK = EG[1] * EG[18] / EG[8] / EG[9];
            RB[56] = RF[56] / EQK;
            RF[57] = 3.2e13;
            EQK = EG[0] * EG[18] / EG[9] / EG[9];
            RB[57] = RF[57] / EQK;
            RF[58] = exp(3.03390713e1 - 3.01930001e2 * Tinv);
            EQK = EG[9] / EG[10];
            RB[58] = RF[58] / EQK;
            RF[59] = 3e13;
            EQK = EG[0] * EG[8] / EG[1] / EG[10];
            RB[59] = RF[59] / EQK;
            RF[60] = 1.5e13;
            EQK = EG[0] * EG[13] / EG[2] / EG[10];
            RB[60] = RF[60] / EQK;
            RF[61] = 1.5e13;
            EQK = EG[1] * EG[15] / EG[2] / EG[10];
            RB[61] = RF[61] / EQK;
            RF[62] = 3e13;
            EQK = EG[1] * EG[16] / EG[4] / EG[10];
            RB[62] = RF[62] / EQK;
            RF[63] = 7e13;
            EQK = EG[1] * EG[11] / EG[0] / EG[10];
            RB[63] = RF[63] / EQK;
            RF[64] = 2.8e13;
            EQK = EG[1] * EG[4] * EG[13] / EG[3] / EG[10] * prefRuT;
            RB[64] = RF[64] / EQK;
            RF[65] = 1.2e13;
            EQK = EG[5] * EG[13] / EG[3] / EG[10];
            RB[65] = RF[65] / EQK;
            RF[66] = 3e13;
            EQK = EG[9] / EG[10];
            RB[66] = RF[66] / EQK;
            RF[67] = 9e12;
            EQK = EG[9] / EG[10];
            RB[67] = RF[67] / EQK;
            RF[68] = 7e12;
            EQK = EG[9] / EG[10];
            RB[68] = RF[68] / EQK;
            RF[69] = 1.4e13;
            EQK = EG[13] * EG[16] / EG[10] / EG[14];
            RB[69] = RF[69] / EQK;
            RF[70] = exp(2.7014835e1 + 4.54e-1 * logT - 1.30836334e3 * Tinv);
            EQK = EG[17] / EG[1] / EG[16] / prefRuT;
            RB[70] = RF[70] / EQK;
            RF[71] = exp(2.38587601e1 + 1.05e0 * logT - 1.64803459e3 * Tinv);
            EQK = EG[0] * EG[15] / EG[1] / EG[16];
            RB[71] = RF[71] / EQK;
            RF[72] = exp(3.12945828e1 - 1.781387e3 * Tinv);
            EQK = EG[4] * EG[15] / EG[2] / EG[16];
            RB[72] = RF[72] / EQK;
            RF[73] = exp(2.19558261e1 + 1.18e0 * logT + 2.2493785e2 * Tinv);
            EQK = EG[5] * EG[15] / EG[4] / EG[16];
            RB[73] = RF[73] / EQK;
            RF[74] = exp(3.22361913e1 - 2.01286667e4 * Tinv);
            EQK = EG[6] * EG[15] / EG[3] / EG[16];
            RB[74] = RF[74] / EQK;
            RF[75] = exp(2.76310211e1 - 4.02573334e3 * Tinv);
            EQK = EG[7] * EG[15] / EG[6] / EG[16];
            RB[75] = RF[75] / EQK;
            RF[76] = exp(3.21806786e1 + 2.59156584e2 * Tinv);
            EQK = EG[1] * EG[25] / EG[8] / EG[16];
            RB[76] = RF[76] / EQK;
            RF[77] = exp(3.70803784e1 - 6.3e-1 * logT - 1.92731984e2 * Tinv);
            EQK = EG[12] / EG[1] / EG[11] / prefRuT;
            RB[77] = RF[77] / EQK;
            RF[78] = 8.43e13;
            EQK = EG[1] * EG[16] / EG[2] / EG[11];
            RB[78] = RF[78] / EQK;
            RF[79] = exp(1.78408622e1 + 1.6e0 * logT - 2.72743434e3 * Tinv);
            EQK = EG[5] * EG[9] / EG[4] / EG[11];
            RB[79] = RF[79] / EQK;
            RF[80] = 2.501e13;
            EQK = EG[5] * EG[10] / EG[4] / EG[11];
            RB[80] = RF[80] / EQK;
            RF[81] = exp(3.10595094e1 - 1.449264e4 * Tinv);
            EQK = EG[2] * EG[17] / EG[3] / EG[11];
            RB[81] = RF[81] / EQK;
            RF[82] = exp(2.43067848e1 - 4.49875701e3 * Tinv);
            EQK = EG[4] * EG[16] / EG[3] / EG[11];
            RB[82] = RF[82] / EQK;
            RF[83] = 1e12;
            EQK = EG[3] * EG[12] / EG[6] / EG[11];
            RB[83] = RF[83] / EQK;
            RF[84] = 1.34e13;
            EQK = EG[4] * EG[17] / EG[6] / EG[11];
            RB[84] = RF[84] / EQK;
            RF[85] = exp(1.01064284e1 + 2.47e0 * logT - 2.60666234e3 * Tinv);
            EQK = EG[6] * EG[12] / EG[7] / EG[11];
            RB[85] = RF[85] / EQK;
            RF[86] = 3e13;
            EQK = EG[1] * EG[20] / EG[8] / EG[11];
            RB[86] = RF[86] / EQK;
            RF[87] = 8.48e12;
            EQK = EG[12] * EG[13] / EG[11] / EG[15];
            RB[87] = RF[87] / EQK;
            RF[88] = 1.8e13;
            EQK = EG[27] / EG[11] / EG[15] / prefRuT;
            RB[88] = RF[88] / EQK;
            RF[89] = exp(8.10772006e0 + 2.81e0 * logT - 2.94884967e3 * Tinv);
            EQK = EG[12] * EG[15] / EG[11] / EG[16];
            RB[89] = RF[89] / EQK;
            RF[90] = 4e13;
            EQK = EG[1] * EG[21] / EG[9] / EG[11];
            RB[90] = RF[90] / EQK;
            RF[91] = exp(3.01159278e1 + 2.86833501e2 * Tinv);
            EQK = EG[1] * EG[21] / EG[10] / EG[11];
            RB[91] = RF[91] / EQK;
            RF[92] = exp(3.75927776e1 - 9.7e-1 * logT - 3.11994334e2 * Tinv);
            EQK = EG[23] / EG[11] / EG[11] / prefRuT;
            RB[92] = RF[92] / EQK;
            RF[93] = exp(2.9238457e1 + 1e-1 * logT - 5.33409668e3 * Tinv);
            EQK = EG[1] * EG[22] / EG[11] / EG[11];
            RB[93] = RF[93] / EQK;
            RF[94] = 5e13;
            EQK = EG[13] * EG[21] / EG[11] / EG[24];
            RB[94] = RF[94] / EQK;
            RF[95] = 2e13;
            EQK = EG[0] * EG[16] / EG[1] / EG[17];
            RB[95] = RF[95] / EQK;
            RF[96] = 3.2e13;
            EQK = EG[4] * EG[11] / EG[1] / EG[17];
            RB[96] = RF[96] / EQK;
            RF[97] = 1.6e13;
            EQK = EG[5] * EG[10] / EG[1] / EG[17];
            RB[97] = RF[97] / EQK;
            RF[98] = 1e13;
            EQK = EG[4] * EG[16] / EG[2] / EG[17];
            RB[98] = RF[98] / EQK;
            RF[99] = 5e12;
            EQK = EG[5] * EG[16] / EG[4] / EG[17];
            RB[99] = RF[99] / EQK;
            RF[100] = exp(-2.84796532e1 + 7.6e0 * logT + 1.77635484e3 * Tinv);
            EQK = EG[6] * EG[16] / EG[3] / EG[17];
            RB[100] = RF[100] / EQK;
            RF[101] = exp(2.03077504e1 + 1.62e0 * logT - 5.45486868e3 * Tinv);
            EQK = EG[0] * EG[11] / EG[1] / EG[12];
            RB[101] = RF[101] / EQK;
            RF[102] = exp(2.07430685e1 + 1.5e0 * logT - 4.32766334e3 * Tinv);
            EQK = EG[4] * EG[11] / EG[2] / EG[12];
            RB[102] = RF[102] / EQK;
            RF[103] = exp(1.84206807e1 + 1.6e0 * logT - 1.570036e3 * Tinv);
            EQK = EG[5] * EG[11] / EG[4] / EG[12];
            RB[103] = RF[103] / EQK;
            RF[104] = 6e13;
            EQK = EG[1] * EG[21] / EG[8] / EG[12];
            RB[104] = RF[104] / EQK;
            RF[105] = exp(1.47156719e1 + 2e0 * logT - 4.16160184e3 * Tinv);
            EQK = EG[11] * EG[11] / EG[9] / EG[12];
            RB[105] = RF[105] / EQK;
            RF[106] = 1.33333333e0 * RF[91];
            EQK = EG[11] * EG[11] / EG[10] / EG[12];
            RB[106] = RF[106] / EQK;
            RF[107] = 1e14;
            EQK = EG[10] * EG[13] / EG[1] / EG[24];
            RB[107] = RF[107] / EQK;
            RF[108] = 1e14;
            EQK = EG[1] * EG[13] * EG[13] / EG[2] / EG[24] * prefRuT;
            RB[108] = RF[108] / EQK;
            RF[109] = exp(2.81010247e1 - 4.29747034e2 * Tinv);
            EQK = EG[4] * EG[13] * EG[13] / EG[3] / EG[24] * prefRuT;
            RB[109] = RF[109] / EQK;
            RF[110] = 5e13;
            EQK = EG[13] * EG[18] / EG[8] / EG[24];
            RB[110] = RF[110] / EQK;
            RF[111] = 3e13;
            EQK = EG[13] * EG[20] / EG[9] / EG[24];
            RB[111] = RF[111] / EQK;
            RF[112] = 1e13;
            EQK = EG[13] * EG[13] * EG[18] / EG[24] / EG[24] * prefRuT;
            RB[112] = RF[112] / EQK;
            RF[113] = exp(3.43156328e1 - 5.2e-1 * logT - 2.55382459e4 * Tinv);
            EQK = EG[19] / EG[18];
            RB[113] = RF[113] / EQK;
            RF[114] = exp(1.97713479e1 + 1.62e0 * logT - 1.86432818e4 * Tinv);
            EQK = EG[1] * EG[18] / EG[20] * prefRuT;
            RB[114] = RF[114] / EQK;
            RF[115] = exp(1.66079019e1 + 2e0 * logT - 9.56111669e2 * Tinv);
            EQK = EG[1] * EG[24] / EG[2] / EG[18];
            RB[115] = RF[115] / EQK;
            RF[116] = 2.5e-1 * RF[115];
            EQK = EG[9] * EG[13] / EG[2] / EG[18];
            RB[116] = RF[116] / EQK;
            RF[117] = exp(-8.4310155e0 + 4.5e0 * logT + 5.03216668e2 * Tinv);
            EQK = EG[1] * EG[25] / EG[4] / EG[18];
            RB[117] = RF[117] / EQK;
            RF[118] = exp(-7.6354939e0 + 4e0 * logT + 1.00643334e3 * Tinv);
            EQK = EG[11] * EG[13] / EG[4] / EG[18];
            RB[118] = RF[118] / EQK;
            RF[119] = exp(1.61180957e1 + 2e0 * logT - 3.01930001e3 * Tinv);
            EQK = EG[13] * EG[20] / EG[15] / EG[18];
            RB[119] = RF[119] / EQK;
            RF[120] = exp(1.27430637e2 - 1.182e1 * logT - 1.79799315e4 * Tinv);
            EQK = EG[28] / EG[11] / EG[18] / prefRuT;
            RB[120] = RF[120] / EQK;
            RF[121] = 1e14;
            EQK = EG[18] / EG[19];
            RB[121] = RF[121] / EQK;
            RF[122] = 1e14;
            EQK = EG[9] * EG[13] / EG[2] / EG[19];
            RB[122] = RF[122] / EQK;
            RF[123] = 2e13;
            EQK = EG[1] * EG[25] / EG[4] / EG[19];
            RB[123] = RF[123] / EQK;
            RF[124] = 1e13;
            EQK = EG[9] * EG[14] / EG[3] / EG[19];
            RB[124] = RF[124] / EQK;
            RF[125] = exp(3.34301138e1 - 6e-2 * logT - 4.27734167e3 * Tinv);
            EQK = EG[26] / EG[1] / EG[25] / prefRuT;
            RB[125] = RF[125] / EQK;
            RF[126] = 5e1 * RF[75];
            EQK = EG[0] * EG[24] / EG[1] / EG[25];
            RB[126] = RF[126] / EQK;
            RF[127] = exp(2.11287309e1 + 1.43e0 * logT - 1.35365284e3 * Tinv);
            EQK = EG[11] * EG[13] / EG[1] / EG[25];
            RB[127] = RF[127] / EQK;
            RF[128] = 1e1 * RF[75];
            EQK = EG[4] * EG[24] / EG[2] / EG[25];
            RB[128] = RF[128] / EQK;
            RF[129] = exp(2.81906369e1 - 6.79342501e2 * Tinv);
            EQK = EG[9] * EG[14] / EG[2] / EG[25];
            RB[129] = RF[129] / EQK;
            RF[130] = exp(2.96459241e1 - 1.00643334e3 * Tinv);
            EQK = EG[5] * EG[24] / EG[4] / EG[25];
            RB[130] = RF[130] / EQK;
            RF[131] = exp(2.94360258e1 + 2.7e-1 * logT - 1.40900667e2 * Tinv);
            EQK = EG[21] / EG[1] / EG[20] / prefRuT;
            RB[131] = RF[131] / EQK;
            RF[132] = 3e13;
            EQK = EG[0] * EG[18] / EG[1] / EG[20];
            RB[132] = RF[132] / EQK;
            RF[133] = 6e13;
            EQK = EG[0] * EG[19] / EG[1] / EG[20];
            RB[133] = RF[133] / EQK;
            RF[134] = 4.8e13;
            EQK = EG[1] * EG[25] / EG[2] / EG[20];
            RB[134] = RF[134] / EQK;
            RF[135] = 4.8e13;
            EQK = EG[11] * EG[13] / EG[2] / EG[20];
            RB[135] = RF[135] / EQK;
            RF[136] = 3.011e13;
            EQK = EG[5] * EG[18] / EG[4] / EG[20];
            RB[136] = RF[136] / EQK;
            RF[137] = exp(1.41081802e1 + 1.61e0 * logT + 1.9293327e2 * Tinv);
            EQK = EG[6] * EG[18] / EG[3] / EG[20];
            RB[137] = RF[137] / EQK;
            RF[138] = exp(2.64270483e1 + 2.9e-1 * logT - 5.53538334e0 * Tinv);
            EQK = EG[2] * EG[26] / EG[3] / EG[20];
            RB[138] = RF[138] / EQK;
            RF[139] = exp(3.83674178e1 - 1.39e0 * logT - 5.08248834e2 * Tinv);
            EQK = EG[15] * EG[16] / EG[3] / EG[20];
            RB[139] = RF[139] / EQK;
            RF[140] = 1e13;
            EQK = EG[4] * EG[26] / EG[6] / EG[20];
            RB[140] = RF[140] / EQK;
            RF[141] = exp(2.32164713e1 + 2.99917134e2 * Tinv);
            EQK = EG[6] * EG[21] / EG[7] / EG[20];
            RB[141] = RF[141] / EQK;
            RF[142] = 9.033e13;
            EQK = EG[13] * EG[21] / EG[15] / EG[20];
            RB[142] = RF[142] / EQK;
            RF[143] = 3.92e11;
            EQK = EG[12] * EG[18] / EG[11] / EG[20];
            RB[143] = RF[143] / EQK;
            RF[144] = 2.5e13;
            EQK = EG[29] / EG[11] / EG[20] / prefRuT;
            RB[144] = RF[144] / EQK;
            RF[145] = exp(5.56675073e1 - 2.83e0 * logT - 9.36888792e3 * Tinv);
            EQK = EG[1] * EG[28] / EG[11] / EG[20];
            RB[145] = RF[145] / EQK;
            RF[146] = exp(9.64601125e1 - 9.147e0 * logT - 2.36008617e4 * Tinv);
            EQK = EG[11] * EG[13] / EG[26] * prefRuT;
            RB[146] = RF[146] / EQK;
            RF[147] = 1e14;
            EQK = EG[27] / EG[1] / EG[26] / prefRuT;
            RB[147] = RF[147] / EQK;
            RF[148] = 9e13;
            EQK = EG[11] * EG[15] / EG[1] / EG[26];
            RB[148] = RF[148] / EQK;
            RF[149] = exp(3.06267534e1 - 2.01286667e3 * Tinv);
            EQK = EG[0] * EG[25] / EG[1] / EG[26];
            RB[149] = RF[149] / EQK;
            RF[150] = RF[149];
            EQK = EG[4] * EG[25] / EG[2] / EG[26];
            RB[150] = RF[150] / EQK;
            RF[151] = 1.33333333e0 * RF[130];
            EQK = EG[5] * EG[25] / EG[4] / EG[26];
            RB[151] = RF[151] / EQK;
            RF[152] = 1.4e11;
            EQK = EG[6] * EG[25] / EG[3] / EG[26];
            RB[152] = RF[152] / EQK;
            RF[153] = 1.8e10;
            EQK = EG[4] * EG[13] * EG[16] / EG[3] / EG[26] * prefRuT;
            RB[153] = RF[153] / EQK;
            RF[154] = exp(2.97104627e1 + 4.4e-1 * logT - 4.46705436e4 * Tinv);
            EQK = EG[0] * EG[19] / EG[21] * prefRuT;
            RB[154] = RF[154] / EQK;
            RF[155] = exp(2.77079822e1 + 4.54e-1 * logT - 9.15854335e2 * Tinv);
            EQK = EG[22] / EG[1] / EG[21] / prefRuT;
            RB[155] = RF[155] / EQK;
            RF[156] = exp(1.77414365e1 + 1.93e0 * logT - 6.51665585e3 * Tinv);
            EQK = EG[0] * EG[20] / EG[1] / EG[21];
            RB[156] = RF[156] / EQK;
            RF[157] = exp(1.65302053e1 + 1.91e0 * logT - 1.88203034e3 * Tinv);
            EQK = EG[4] * EG[20] / EG[2] / EG[21];
            RB[157] = RF[157] / EQK;
            RF[158] = exp(1.67704208e1 + 1.83e0 * logT - 1.10707667e2 * Tinv);
            EQK = EG[11] * EG[15] / EG[2] / EG[21];
            RB[158] = RF[158] / EQK;
            RF[159] = 2e-2 * RF[158];
            EQK = EG[9] * EG[16] / EG[2] / EG[21];
            RB[159] = RF[159] / EQK;
            RF[160] = exp(1.50964444e1 + 2e0 * logT - 1.25804167e3 * Tinv);
            EQK = EG[5] * EG[20] / EG[4] / EG[21];
            RB[160] = RF[160] / EQK;
            RF[161] = exp(3.13734413e1 - 3.05955734e4 * Tinv);
            EQK = EG[6] * EG[20] / EG[3] / EG[21];
            RB[161] = RF[161] / EQK;
            RF[162] = exp(2.83241683e1 - 7.04503335e3 * Tinv);
            EQK = EG[4] * EG[27] / EG[6] / EG[21];
            RB[162] = RF[162] / EQK;
            RF[163] = exp(1.61180957e1 + 2e0 * logT - 4.02573334e3 * Tinv);
            EQK = EG[13] * EG[22] / EG[15] / EG[21];
            RB[163] = RF[163] / EQK;
            RF[164] = exp(3.06267534e1 - 3.01930001e3 * Tinv);
            EQK = EG[1] * EG[28] / EG[9] / EG[21];
            RB[164] = RF[164] / EQK;
            RF[165] = 5e13;
            EQK = EG[12] * EG[19] / EG[10] / EG[21];
            RB[165] = RF[165] / EQK;
            RF[166] = 5e13;
            EQK = EG[1] * EG[28] / EG[10] / EG[21];
            RB[166] = RF[166] / EQK;
            RF[167] = exp(1.23327053e1 + 2e0 * logT - 4.62959334e3 * Tinv);
            EQK = EG[12] * EG[20] / EG[11] / EG[21];
            RB[167] = RF[167] / EQK;
            RF[168] = exp(2.65223585e1 - 3.87476834e3 * Tinv);
            EQK = EG[30] / EG[11] / EG[21] / prefRuT;
            RB[168] = RF[168] / EQK;
            RF[169] = exp(4.07945264e1 - 9.9e-1 * logT - 7.95082335e2 * Tinv);
            EQK = EG[23] / EG[1] / EG[22] / prefRuT;
            RB[169] = RF[169] / EQK;
            RF[170] = 2e12;
            EQK = EG[0] * EG[21] / EG[1] / EG[22];
            RB[170] = RF[170] / EQK;
            RF[171] = 1.604e13;
            EQK = EG[11] * EG[16] / EG[2] / EG[22];
            RB[171] = RF[171] / EQK;
            RF[172] = 8.02e13;
            EQK = EG[1] * EG[27] / EG[2] / EG[22];
            RB[172] = RF[172] / EQK;
            RF[173] = 2e10;
            EQK = EG[6] * EG[21] / EG[3] / EG[22];
            RB[173] = RF[173] / EQK;
            RF[174] = 3e11;
            EQK = EG[3] * EG[23] / EG[6] / EG[22];
            RB[174] = RF[174] / EQK;
            RF[175] = 3e11;
            EQK = EG[7] * EG[21] / EG[6] / EG[22];
            RB[175] = RF[175] / EQK;
            RF[176] = 2.4e13;
            EQK = EG[4] * EG[11] * EG[16] / EG[6] / EG[22] * prefRuT;
            RB[176] = RF[176] / EQK;
            RF[177] = exp(2.28865889e1 - 4.90133034e2 * Tinv);
            EQK = EG[6] * EG[23] / EG[7] / EG[22];
            RB[177] = RF[177] / EQK;
            RF[178] = 1.2e14;
            EQK = EG[13] * EG[23] / EG[15] / EG[22];
            RB[178] = RF[178] / EQK;
            RF[179] = exp(1.85604427e1 + 1.9e0 * logT - 3.78922151e3 * Tinv);
            EQK = EG[0] * EG[22] / EG[1] / EG[23];
            RB[179] = RF[179] / EQK;
            RF[180] = exp(1.83130955e1 + 1.92e0 * logT - 2.86330284e3 * Tinv);
            EQK = EG[4] * EG[22] / EG[2] / EG[23];
            RB[180] = RF[180] / EQK;
            RF[181] = exp(1.50796373e1 + 2.12e0 * logT - 4.37798501e2 * Tinv);
            EQK = EG[5] * EG[22] / EG[4] / EG[23];
            RB[181] = RF[181] / EQK;
            RF[182] = exp(3.13199006e1 + 2.76769167e2 * Tinv);
            EQK = EG[11] * EG[22] / EG[10] / EG[23];
            RB[182] = RF[182] / EQK;
            RF[183] = exp(1.56303353e1 + 1.74e0 * logT - 5.25861418e3 * Tinv);
            EQK = EG[12] * EG[22] / EG[11] / EG[23];
            RB[183] = RF[183] / EQK;
            RF[184] = 2e14;
            EQK = EG[29] / EG[1] / EG[28] / prefRuT;
            RB[184] = RF[184] / EQK;
            RF[185] = 2.66666667e0 * RF[130];
            EQK = EG[12] * EG[19] / EG[1] / EG[28];
            RB[185] = RF[185] / EQK;
            RF[186] = 2.66e12;
            EQK = EG[3] * EG[29] / EG[6] / EG[28];
            RB[186] = RF[186] / EQK;
            RF[187] = 6.6e12;
            EQK = EG[4] * EG[16] * EG[20] / EG[6] / EG[28] * prefRuT;
            RB[187] = RF[187] / EQK;
            RF[188] = 6e13;
            EQK = EG[13] * EG[29] / EG[15] / EG[28];
            RB[188] = RF[188] / EQK;
            RF[189] = exp(3.02187852e1 - 1.64083859e3 * Tinv);
            EQK = EG[30] / EG[1] / EG[29] / prefRuT;
            RB[189] = RF[189] / EQK;
            RF[190] = exp(5.11268757e1 - 2.39e0 * logT - 5.62596234e3 * Tinv);
            EQK = EG[11] * EG[21] / EG[1] / EG[29];
            RB[190] = RF[190] / EQK;
            RF[191] = exp(1.20435537e1 + 2.5e0 * logT - 1.2530095e3 * Tinv);
            EQK = EG[0] * EG[28] / EG[1] / EG[29];
            RB[191] = RF[191] / EQK;
            RF[192] = exp(1.86030023e1 + 1.65e0 * logT - 1.6455185e2 * Tinv);
            EQK = EG[1] * EG[11] * EG[25] / EG[2] / EG[29] * prefRuT;
            RB[192] = RF[192] / EQK;
            RF[193] = exp(1.73708586e1 + 1.65e0 * logT + 4.89126601e2 * Tinv);
            EQK = EG[15] * EG[22] / EG[2] / EG[29];
            RB[193] = RF[193] / EQK;
            RF[194] = exp(2.59162227e1 + 7e-1 * logT - 2.95891401e3 * Tinv);
            EQK = EG[4] * EG[28] / EG[2] / EG[29];
            RB[194] = RF[194] / EQK;
            RF[195] = exp(1.49469127e1 + 2e0 * logT + 1.49958567e2 * Tinv);
            EQK = EG[5] * EG[28] / EG[4] / EG[29];
            RB[195] = RF[195] / EQK;
            RF[196] = exp(9.16951838e0 + 2.6e0 * logT - 6.99974385e3 * Tinv);
            EQK = EG[7] * EG[28] / EG[6] / EG[29];
            RB[196] = RF[196] / EQK;
            RF[197] = exp(7.8845736e-1 + 3.5e0 * logT - 2.85575459e3 * Tinv);
            EQK = EG[12] * EG[28] / EG[11] / EG[29];
            RB[197] = RF[197] / EQK;
            RF[198] = exp(5.65703751e1 - 2.92e0 * logT - 6.29272443e3 * Tinv);
            EQK = EG[11] * EG[22] / EG[1] / EG[30];
            RB[198] = RF[198] / EQK;
            RF[199] = 1.8e12;
            EQK = EG[0] * EG[29] / EG[1] / EG[30];
            RB[199] = RF[199] / EQK;
            RF[200] = 9.6e13;
            EQK = EG[16] * EG[22] / EG[2] / EG[30];
            RB[200] = RF[200] / EQK;
            RF[201] = 2.4e13;
            EQK = EG[5] * EG[29] / EG[4] / EG[30];
            RB[201] = RF[201] / EQK;
            RF[202] = 9e10;
            EQK = EG[6] * EG[29] / EG[3] / EG[30];
            RB[202] = RF[202] / EQK;
            RF[203] = 2.4e13;
            EQK = EG[4] * EG[16] * EG[22] / EG[6] / EG[30] * prefRuT;
            RB[203] = RF[203] / EQK;
            RF[204] = 1.1e13;
            EQK = EG[12] * EG[29] / EG[11] / EG[30];
            RB[204] = RF[204] / EQK;
            RF[205] = exp(7.50436995e1 - 5.22e0 * logT - 9.93701954e3 * Tinv);
            EQK = EG[11] * EG[28] / EG[20] / EG[22];
            RB[205] = RF[205] / EQK;
          } // end scope

          { // start scope
            double CTB, PR, PCOR, PRLOG, RKLOW;
            double FCENT, XN, CPRLOG, FLOG, FCLOG, FC;
            RF[0] = RF[0] * cs[1] * cs[3];
            RB[0] = RB[0] * cs[2] * cs[4];
            RF[1] = RF[1] * cs[0] * cs[2];
            RB[1] = RB[1] * cs[1] * cs[4];
            RF[2] = RF[2] * cs[0] * cs[4];
            RB[2] = RB[2] * cs[1] * cs[5];
            RF[3] = RF[3] * cs[4] * cs[4];
            RB[3] = RB[3] * cs[2] * cs[5];
            CTB = CTOT - cs[0] - cs[5] + cs[9] - cs[11] + 2e0 * cs[15] +
                  2e0 * cs[13] + 2e0 * cs[14];
            RF[4] = RF[4] * CTB * cs[1] * cs[1];
            RB[4] = RB[4] * CTB * cs[0];
            RF[5] = RF[5] * cs[0] * cs[1] * cs[1];
            RB[5] = RB[5] * cs[0] * cs[0];
            RF[6] = RF[6] * cs[1] * cs[1] * cs[5];
            RB[6] = RB[6] * cs[0] * cs[5];
            RF[7] = RF[7] * cs[1] * cs[1] * cs[11];
            RB[7] = RB[7] * cs[0] * cs[11];

            CTB = CTOT - 2.7e-1 * cs[0] + 2.65e0 * cs[5] + cs[9] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RF[8] = RF[8] * CTB * cs[1] * cs[4];
            RB[8] = RB[8] * CTB * cs[5];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RF[9] = RF[9] * CTB * cs[1] * cs[2];
            RB[9] = RB[9] * CTB * cs[4];
            CTB = CTOT + 1.4e0 * cs[0] + 1.44e1 * cs[5] + cs[9] +
                  7.5e-1 * cs[10] + 2.6e0 * cs[11] + 2e0 * cs[15] +
                  2e0 * cs[13] + 2e0 * cs[14];
            RF[10] = RF[10] * CTB * cs[2] * cs[2];
            RB[10] = RB[10] * CTB * cs[3];
            CTB = CTOT - cs[3] - cs[5] - 2.5e-1 * cs[10] + 5e-1 * cs[11] +
                  5e-1 * cs[15] - cs[21] + 2e0 * cs[13] + 2e0 * cs[14];
            RF[11] = RF[11] * CTB * cs[1] * cs[3];
            RB[11] = RB[11] * CTB * cs[6];
            RF[12] = RF[12] * cs[1] * cs[3] * cs[3];
            RB[12] = RB[12] * cs[3] * cs[6];
            RF[13] = RF[13] * cs[1] * cs[3] * cs[5];
            RB[13] = RB[13] * cs[5] * cs[6];
            RF[14] = RF[14] * cs[1] * cs[3] * cs[21];
            RB[14] = RB[14] * cs[6] * cs[21];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(4.22794408e1 - 9e-1 * logT + 8.55468335e2 * Tinv);
            PR = RKLOW * CTB / RF[15];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.654e-1 * exp(-T / 9.4e1) + 7.346e-1 * exp(-T / 1.756e3) +
                    exp(-5.182e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[15] = RF[15] * PCOR;
            RB[15] = RB[15] * PCOR;
            RF[15] = RF[15] * cs[4] * cs[4];
            RB[15] = RB[15] * cs[7];
            RF[16] = RF[16] * cs[1] * cs[6];
            RB[16] = RB[16] * cs[2] * cs[5];
            RF[17] = RF[17] * cs[1] * cs[6];
            RB[17] = RB[17] * cs[0] * cs[3];
            RF[18] = RF[18] * cs[1] * cs[6];
            RB[18] = RB[18] * cs[4] * cs[4];
            RF[19] = RF[19] * cs[2] * cs[6];
            RB[19] = RB[19] * cs[3] * cs[4];
            RF[20] = RF[20] * cs[4] * cs[6];
            RB[20] = RB[20] * cs[3] * cs[5];
            RF[21] = RF[21] * cs[6] * cs[6];
            RB[21] = RB[21] * cs[3] * cs[7];
            RF[22] = RF[22] * cs[6] * cs[6];
            RB[22] = RB[22] * cs[3] * cs[7];
            RF[23] = RF[23] * cs[1] * cs[7];
            RB[23] = RB[23] * cs[0] * cs[6];
            RF[24] = RF[24] * cs[1] * cs[7];
            RB[24] = RB[24] * cs[4] * cs[5];
            RF[25] = RF[25] * cs[2] * cs[7];
            RB[25] = RB[25] * cs[4] * cs[6];
            RF[26] = RF[26] * cs[4] * cs[7];
            RB[26] = RB[26] * cs[5] * cs[6];
            RF[27] = RF[27] * cs[4] * cs[7];
            RB[27] = RB[27] * cs[5] * cs[6];
            CTB = CTOT + cs[0] + 5e0 * cs[3] + 5e0 * cs[5] + cs[9] +
                  5e-1 * cs[10] + 2.5e0 * cs[11] + 2e0 * cs[15] + 2e0 * cs[13] +
                  2e0 * cs[14];
            RF[28] = RF[28] * CTB * cs[2] * cs[10];
            RB[28] = RB[28] * CTB * cs[11];
            RF[29] = RF[29] * cs[4] * cs[10];
            RB[29] = RB[29] * cs[1] * cs[11];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.37931383e1 - 3.42e0 * logT - 4.24463259e4 * Tinv);
            PR = RKLOW * CTB / RF[30];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 6.8e-2 * exp(-T / 1.97e2) + 9.32e-1 * exp(-T / 1.54e3) +
                    exp(-1.03e4 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[30] = RF[30] * PCOR;
            RB[30] = RB[30] * PCOR;
            RF[30] = RF[30] * cs[0] * cs[10];
            RB[30] = RB[30] * cs[12];
            RF[31] = RF[31] * cs[3] * cs[10];
            RB[31] = RB[31] * cs[2] * cs[11];
            RF[32] = RF[32] * cs[6] * cs[10];
            RB[32] = RB[32] * cs[4] * cs[11];
            RF[33] = RF[33] * cs[2];
            RB[33] = RB[33] * cs[1] * cs[10];
            RF[34] = RF[34] * cs[4];
            RB[34] = RB[34] * cs[1];
            RF[35] = RF[35] * cs[0];
            RB[35] = RB[35] * cs[1];
            RF[36] = RF[36] * cs[5];
            RB[36] = RB[36] * cs[1] * cs[12];
            RF[37] = RF[37] * cs[3];
            RB[37] = RB[37] * cs[2];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.54619238e1 - 3.74e0 * logT - 9.74227469e2 * Tinv);
            PR = RKLOW * CTB / RF[38];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 4.243e-1 * exp(-T / 2.37e2) + 5.757e-1 * exp(-T / 1.652e3) +
                    exp(-5.069e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[38] = RF[38] * PCOR;
            RB[38] = RB[38] * PCOR;
            RF[38] = RF[38] * cs[10];
            RB[38] = RB[38] * cs[16];
            RF[39] = RF[39] * cs[11];
            RB[39] = RB[39] * cs[10];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(5.55621468e1 - 2.57e0 * logT - 7.17083751e2 * Tinv);
            PR = RKLOW * CTB / RF[40];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.176e-1 * exp(-T / 2.71e2) + 7.824e-1 * exp(-T / 2.755e3) +
                    exp(-6.57e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[40] = RF[40] * PCOR;
            RB[40] = RB[40] * PCOR;
            RF[40] = RF[40] * cs[1];
            RB[40] = RB[40] * cs[12];
            RF[41] = RF[41] * cs[1];
            RB[41] = RB[41] * cs[0] * cs[10];
            RF[42] = RF[42] * cs[2];
            RB[42] = RB[42] * cs[4] * cs[10];
            RF[43] = RF[43] * cs[2];
            RB[43] = RB[43] * cs[1] * cs[11];
            RF[44] = RF[44] * cs[4];
            RB[44] = RB[44] * cs[5] * cs[10];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RF[45] = RF[45] * CTB;
            RB[45] = RB[45] * CTB * cs[1] * cs[10];
            RF[46] = RF[46] * cs[3];
            RB[46] = RB[46] * cs[6] * cs[10];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.33329483e1 - 3.14e0 * logT - 6.18956501e2 * Tinv);
            PR = RKLOW * CTB / RF[47];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 3.2e-1 * exp(-T / 7.8e1) + 6.8e-1 * exp(-T / 1.995e3) +
                    exp(-5.59e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[47] = RF[47] * PCOR;
            RB[47] = RB[47] * PCOR;
            RF[47] = RF[47] * cs[1];
            RB[47] = RB[47] * cs[8];
            RF[48] = RF[48] * cs[0];
            RB[48] = RB[48] * cs[1] * cs[8];
            RF[49] = RF[49] * cs[2];
            RB[49] = RB[49] * cs[1];
            RF[50] = RF[50] * cs[3];
            RB[50] = RB[50] * cs[4];
            RF[51] = RF[51] * cs[3];
            RB[51] = RB[51] * cs[1] * cs[1] * cs[11];
            RF[52] = RF[52] * cs[4];
            RB[52] = RB[52] * cs[1] * cs[12];
            RF[53] = RF[53] * cs[4];
            RB[53] = RB[53] * cs[5];
            RF[54] = RF[54] * cs[6];
            RB[54] = RB[54] * cs[4] * cs[12];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(7.69748493e1 - 5.11e0 * logT - 3.57032226e3 * Tinv);
            PR = RKLOW * CTB / RF[55];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 4.093e-1 * exp(-T / 2.75e2) + 5.907e-1 * exp(-T / 1.226e3) +
                    exp(-5.185e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[55] = RF[55] * PCOR;
            RB[55] = RB[55] * PCOR;
            RF[55] = RF[55] * cs[10];
            RB[55] = RB[55] * cs[17];
            RB[56] = RB[56] * cs[1] * cs[13];
            RB[57] = RB[57] * cs[0] * cs[13];
            RF[58] = RF[58] * cs[21];
            RB[58] = RB[58] * cs[21];
            RF[59] = RF[59] * cs[1];
            RB[59] = RB[59] * cs[0];
            RF[60] = RF[60] * cs[2];
            RB[60] = RB[60] * cs[0] * cs[10];
            RF[61] = RF[61] * cs[2];
            RB[61] = RB[61] * cs[1];
            RF[62] = RF[62] * cs[4];
            RB[62] = RB[62] * cs[1] * cs[12];
            RF[63] = RF[63] * cs[0];
            RB[63] = RB[63] * cs[1] * cs[8];
            RF[64] = RF[64] * cs[3];
            RB[64] = RB[64] * cs[1] * cs[4] * cs[10];
            RF[65] = RF[65] * cs[3];
            RB[65] = RB[65] * cs[5] * cs[10];
            RF[66] = RF[66] * cs[5];
            RB[66] = RB[66] * cs[5];
            RF[67] = RF[67] * cs[10];
            RB[67] = RB[67] * cs[10];
            RF[68] = RF[68] * cs[11];
            RB[68] = RB[68] * cs[11];
            RF[69] = RF[69] * cs[11];
            RB[69] = RB[69] * cs[10] * cs[12];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.98660102e1 - 4.8e0 * logT - 2.79788467e3 * Tinv);
            PR = RKLOW * CTB / RF[70];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.42e-1 * exp(-T / 9.4e1) + 7.58e-1 * exp(-T / 1.555e3) +
                    exp(-4.2e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[70] = RF[70] * PCOR;
            RB[70] = RB[70] * PCOR;
            RF[70] = RF[70] * cs[1] * cs[12];
            RF[71] = RF[71] * cs[1] * cs[12];
            RB[71] = RB[71] * cs[0];
            RF[72] = RF[72] * cs[2] * cs[12];
            RB[72] = RB[72] * cs[4];
            RF[73] = RF[73] * cs[4] * cs[12];
            RB[73] = RB[73] * cs[5];
            RF[74] = RF[74] * cs[3] * cs[12];
            RB[74] = RB[74] * cs[6];
            RF[75] = RF[75] * cs[6] * cs[12];
            RB[75] = RB[75] * cs[7];
            RF[76] = RF[76] * cs[12];
            RB[76] = RB[76] * cs[1] * cs[17];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(7.68923562e1 - 4.76e0 * logT - 1.22784867e3 * Tinv);
            PR = RKLOW * CTB / RF[77];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.17e-1 * exp(-T / 7.4e1) + 7.83e-1 * exp(-T / 2.941e3) +
                    exp(-6.964e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[77] = RF[77] * PCOR;
            RB[77] = RB[77] * PCOR;
            RF[77] = RF[77] * cs[1] * cs[8];
            RB[77] = RB[77] * cs[9];
            RF[78] = RF[78] * cs[2] * cs[8];
            RB[78] = RB[78] * cs[1] * cs[12];
            RF[79] = RF[79] * cs[4] * cs[8];
            RB[79] = RB[79] * cs[5];
            RF[80] = RF[80] * cs[4] * cs[8];
            RB[80] = RB[80] * cs[5];
            RF[81] = RF[81] * cs[3] * cs[8];
            RB[81] = RB[81] * cs[2];
            RF[82] = RF[82] * cs[3] * cs[8];
            RB[82] = RB[82] * cs[4] * cs[12];
            RF[83] = RF[83] * cs[6] * cs[8];
            RB[83] = RB[83] * cs[3] * cs[9];
            RF[84] = RF[84] * cs[6] * cs[8];
            RB[84] = RB[84] * cs[4];
            RF[85] = RF[85] * cs[7] * cs[8];
            RB[85] = RB[85] * cs[6] * cs[9];
            RF[86] = RF[86] * cs[8];
            RB[86] = RB[86] * cs[1];
            RF[87] = RF[87] * cs[8];
            RB[87] = RB[87] * cs[9] * cs[10];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(1.11312542e2 - 9.588e0 * logT - 2.566405e3 * Tinv);
            PR = RKLOW * CTB / RF[88];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 3.827e-1 * exp(-T / 1.3076e1) +
                    6.173e-1 * exp(-T / 2.078e3) + exp(-5.093e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[88] = RF[88] * PCOR;
            RB[88] = RB[88] * PCOR;
            RF[88] = RF[88] * cs[8];
            RB[88] = RB[88] * cs[18];
            RF[89] = RF[89] * cs[8] * cs[12];
            RB[89] = RB[89] * cs[9];
            RF[90] = RF[90] * cs[8];
            RB[90] = RB[90] * cs[1] * cs[14];
            RF[91] = RF[91] * cs[8];
            RB[91] = RB[91] * cs[1] * cs[14];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(1.15700234e2 - 9.67e0 * logT - 3.13000767e3 * Tinv);
            PR = RKLOW * CTB / RF[92];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 4.675e-1 * exp(-T / 1.51e2) + 5.325e-1 * exp(-T / 1.038e3) +
                    exp(-4.97e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[92] = RF[92] * PCOR;
            RB[92] = RB[92] * PCOR;
            RF[92] = RF[92] * cs[8] * cs[8];
            RB[92] = RB[92] * cs[15];
            RF[93] = RF[93] * cs[8] * cs[8];
            RB[93] = RB[93] * cs[1];
            RF[94] = RF[94] * cs[8] * cs[16];
            RB[94] = RB[94] * cs[10] * cs[14];
            RF[95] = RF[95] * cs[1];
            RB[95] = RB[95] * cs[0] * cs[12];
            RF[96] = RF[96] * cs[1];
            RB[96] = RB[96] * cs[4] * cs[8];
            RF[97] = RF[97] * cs[1];
            RB[97] = RB[97] * cs[5];
            RF[98] = RF[98] * cs[2];
            RB[98] = RB[98] * cs[4] * cs[12];
            RF[99] = RF[99] * cs[4];
            RB[99] = RB[99] * cs[5] * cs[12];
            RF[100] = RF[100] * cs[3];
            RB[100] = RB[100] * cs[6] * cs[12];
            RF[101] = RF[101] * cs[1] * cs[9];
            RB[101] = RB[101] * cs[0] * cs[8];
            RF[102] = RF[102] * cs[2] * cs[9];
            RB[102] = RB[102] * cs[4] * cs[8];
            RF[103] = RF[103] * cs[4] * cs[9];
            RB[103] = RB[103] * cs[5] * cs[8];
            RF[104] = RF[104] * cs[9];
            RB[104] = RB[104] * cs[1] * cs[14];
            RF[105] = RF[105] * cs[9];
            RB[105] = RB[105] * cs[8] * cs[8];
            RF[106] = RF[106] * cs[9];
            RB[106] = RB[106] * cs[8] * cs[8];
            RF[107] = RF[107] * cs[1] * cs[16];
            RB[107] = RB[107] * cs[10];
            RF[108] = RF[108] * cs[2] * cs[16];
            RB[108] = RB[108] * cs[1] * cs[10] * cs[10];
            RF[109] = RF[109] * cs[3] * cs[16];
            RB[109] = RB[109] * cs[4] * cs[10] * cs[10];
            RF[110] = RF[110] * cs[16];
            RB[110] = RB[110] * cs[10] * cs[13];
            RF[111] = RF[111] * cs[16];
            RB[111] = RB[111] * cs[10];
            RF[112] = RF[112] * cs[16] * cs[16];
            RB[112] = RB[112] * cs[10] * cs[10] * cs[13];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 1.5e0 * cs[13] + 1.5e0 * cs[14];
            RKLOW = exp(3.54348644e1 - 6.4e-1 * logT - 2.50098684e4 * Tinv);
            PR = RKLOW * CTB / RF[113];
            PCOR = PR / (1.0 + PR);
            RF[113] = RF[113] * PCOR;
            RB[113] = RB[113] * PCOR;
            RF[113] = RF[113] * cs[13];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.3111756e1 - 3.4e0 * logT - 1.80145126e4 * Tinv);
            PR = RKLOW * CTB / RF[114];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = -9.816e-1 * exp(-T / 5.3837e3) +
                    1.9816e0 * exp(-T / 4.2932e0) + exp(7.95e-2 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[114] = RF[114] * PCOR;
            RB[114] = RB[114] * PCOR;
            RB[114] = RB[114] * cs[1] * cs[13];
            RF[115] = RF[115] * cs[2] * cs[13];
            RB[115] = RB[115] * cs[1] * cs[16];
            RF[116] = RF[116] * cs[2] * cs[13];
            RB[116] = RB[116] * cs[10];
            RF[117] = RF[117] * cs[4] * cs[13];
            RB[117] = RB[117] * cs[1] * cs[17];
            RF[118] = RF[118] * cs[4] * cs[13];
            RB[118] = RB[118] * cs[8] * cs[10];
            RF[119] = RF[119] * cs[13];
            RB[119] = RB[119] * cs[10];
            CTB = CTOT;
            RF[120] = RF[120] * CTB * cs[8] * cs[13];
            RB[120] = RB[120] * CTB * cs[19];
            RF[121] = RF[121] * cs[1];
            RB[121] = RB[121] * cs[1] * cs[13];
            RF[122] = RF[122] * cs[2];
            RB[122] = RB[122] * cs[10];
            RF[123] = RF[123] * cs[4];
            RB[123] = RB[123] * cs[1] * cs[17];
            RF[124] = RF[124] * cs[3];
            RB[124] = RB[124] * cs[11];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(9.57409899e1 - 7.64e0 * logT - 5.98827834e3 * Tinv);
            PR = RKLOW * CTB / RF[125];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 6.63e-1 * exp(-T / 1.707e3) + 3.37e-1 * exp(-T / 3.2e3) +
                    exp(-4.131e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[125] = RF[125] * PCOR;
            RB[125] = RB[125] * PCOR;
            RF[125] = RF[125] * cs[1] * cs[17];
            RF[126] = RF[126] * cs[1] * cs[17];
            RB[126] = RB[126] * cs[0] * cs[16];
            RF[127] = RF[127] * cs[1] * cs[17];
            RB[127] = RB[127] * cs[8] * cs[10];
            RF[128] = RF[128] * cs[2] * cs[17];
            RB[128] = RB[128] * cs[4] * cs[16];
            RF[129] = RF[129] * cs[2] * cs[17];
            RB[129] = RB[129] * cs[11];
            RF[130] = RF[130] * cs[4] * cs[17];
            RB[130] = RB[130] * cs[5] * cs[16];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(6.9414025e1 - 3.86e0 * logT - 1.67067934e3 * Tinv);
            PR = RKLOW * CTB / RF[131];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.18e-1 * exp(-T / 2.075e2) + 7.82e-1 * exp(-T / 2.663e3) +
                    exp(-6.095e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[131] = RF[131] * PCOR;
            RB[131] = RB[131] * PCOR;
            RF[131] = RF[131] * cs[1];
            RB[131] = RB[131] * cs[14];
            RF[132] = RF[132] * cs[1];
            RB[132] = RB[132] * cs[0] * cs[13];
            RF[133] = RF[133] * cs[1];
            RB[133] = RB[133] * cs[0];
            RF[134] = RF[134] * cs[2];
            RB[134] = RB[134] * cs[1] * cs[17];
            RF[135] = RF[135] * cs[2];
            RB[135] = RB[135] * cs[8] * cs[10];
            RF[136] = RF[136] * cs[4];
            RB[136] = RB[136] * cs[5] * cs[13];
            RF[137] = RF[137] * cs[3];
            RB[137] = RB[137] * cs[6] * cs[13];
            RF[138] = RF[138] * cs[3];
            RB[138] = RB[138] * cs[2];
            RF[139] = RF[139] * cs[3];
            RB[139] = RB[139] * cs[12];
            RF[140] = RF[140] * cs[6];
            RB[140] = RB[140] * cs[4];
            RF[141] = RF[141] * cs[7];
            RB[141] = RB[141] * cs[6] * cs[14];
            RB[142] = RB[142] * cs[10] * cs[14];
            RF[143] = RF[143] * cs[8];
            RB[143] = RB[143] * cs[9] * cs[13];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(1.35001549e2 - 1.194e1 * logT - 4.9163262e3 * Tinv);
            PR = RKLOW * CTB / RF[144];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 8.25e-1 * exp(-T / 1.3406e3) + 1.75e-1 * exp(-T / 6e4) +
                    exp(-1.01398e4 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[144] = RF[144] * PCOR;
            RB[144] = RB[144] * PCOR;
            RF[144] = RF[144] * cs[8];
            RB[144] = RB[144] * cs[20];
            RF[145] = RF[145] * cs[8];
            RB[145] = RB[145] * cs[1] * cs[19];
            RB[146] = RB[146] * cs[8] * cs[10];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(9.14494773e1 - 7.297e0 * logT - 2.36511834e3 * Tinv);
            PR = RKLOW * CTB / RF[147];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 4.5e-1 * exp(-T / 8.9e3) + 5.5e-1 * exp(-T / 4.35e3) +
                    exp(-7.244e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[147] = RF[147] * PCOR;
            RB[147] = RB[147] * PCOR;
            RF[147] = RF[147] * cs[1];
            RB[147] = RB[147] * cs[18];
            RF[148] = RF[148] * cs[1];
            RB[148] = RB[148] * cs[8];
            RF[149] = RF[149] * cs[1];
            RB[149] = RB[149] * cs[0] * cs[17];
            RF[150] = RF[150] * cs[2];
            RB[150] = RB[150] * cs[4] * cs[17];
            RF[151] = RF[151] * cs[4];
            RB[151] = RB[151] * cs[5] * cs[17];
            RF[152] = RF[152] * cs[3];
            RB[152] = RB[152] * cs[6] * cs[17];
            RF[153] = RF[153] * cs[3];
            RB[153] = RB[153] * cs[4] * cs[10] * cs[12];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(1.17075165e2 - 9.31e0 * logT - 5.02512164e4 * Tinv);
            PR = RKLOW * CTB / RF[154];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.655e-1 * exp(-T / 1.8e2) + 7.345e-1 * exp(-T / 1.035e3) +
                    exp(-5.417e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[154] = RF[154] * PCOR;
            RB[154] = RB[154] * PCOR;
            RF[154] = RF[154] * cs[14];
            RB[154] = RB[154] * cs[0];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(9.68908955e1 - 7.62e0 * logT - 3.50742017e3 * Tinv);
            PR = RKLOW * CTB / RF[155];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 2.47e-2 * exp(-T / 2.1e2) + 9.753e-1 * exp(-T / 9.84e2) +
                    exp(-4.374e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[155] = RF[155] * PCOR;
            RB[155] = RB[155] * PCOR;
            RF[155] = RF[155] * cs[1] * cs[14];
            RF[156] = RF[156] * cs[1] * cs[14];
            RB[156] = RB[156] * cs[0];
            RF[157] = RF[157] * cs[2] * cs[14];
            RB[157] = RB[157] * cs[4];
            RF[158] = RF[158] * cs[2] * cs[14];
            RB[158] = RB[158] * cs[8];
            RF[159] = RF[159] * cs[2] * cs[14];
            RB[159] = RB[159] * cs[12];
            RF[160] = RF[160] * cs[4] * cs[14];
            RB[160] = RB[160] * cs[5];
            RF[161] = RF[161] * cs[3] * cs[14];
            RB[161] = RB[161] * cs[6];
            RF[162] = RF[162] * cs[6] * cs[14];
            RB[162] = RB[162] * cs[4] * cs[18];
            RF[163] = RF[163] * cs[14];
            RB[163] = RB[163] * cs[10];
            RF[164] = RF[164] * cs[14];
            RB[164] = RB[164] * cs[1] * cs[19];
            RF[165] = RF[165] * cs[14];
            RB[165] = RB[165] * cs[9];
            RF[166] = RF[166] * cs[14];
            RB[166] = RB[166] * cs[1] * cs[19];
            RF[167] = RF[167] * cs[8] * cs[14];
            RB[167] = RB[167] * cs[9];
            RF[168] = RF[168] * cs[8] * cs[14];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(9.50941235e1 - 7.08e0 * logT - 3.36400342e3 * Tinv);
            PR = RKLOW * CTB / RF[169];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 1.578e-1 * exp(-T / 1.25e2) + 8.422e-1 * exp(-T / 2.219e3) +
                    exp(-6.882e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[169] = RF[169] * PCOR;
            RB[169] = RB[169] * PCOR;
            RF[169] = RF[169] * cs[1];
            RB[169] = RB[169] * cs[15];
            RF[170] = RF[170] * cs[1];
            RB[170] = RB[170] * cs[0] * cs[14];
            RF[171] = RF[171] * cs[2];
            RB[171] = RB[171] * cs[8] * cs[12];
            RF[172] = RF[172] * cs[2];
            RB[172] = RB[172] * cs[1] * cs[18];
            RF[173] = RF[173] * cs[3];
            RB[173] = RB[173] * cs[6] * cs[14];
            RF[174] = RF[174] * cs[6];
            RB[174] = RB[174] * cs[3] * cs[15];
            RF[175] = RF[175] * cs[6];
            RB[175] = RB[175] * cs[7] * cs[14];
            RF[176] = RF[176] * cs[6];
            RB[176] = RB[176] * cs[4] * cs[8] * cs[12];
            RF[177] = RF[177] * cs[7];
            RB[177] = RB[177] * cs[6] * cs[15];
            RB[178] = RB[178] * cs[10] * cs[15];
            RF[179] = RF[179] * cs[1] * cs[15];
            RB[179] = RB[179] * cs[0];
            RF[180] = RF[180] * cs[2] * cs[15];
            RB[180] = RB[180] * cs[4];
            RF[181] = RF[181] * cs[4] * cs[15];
            RB[181] = RB[181] * cs[5];
            RF[182] = RF[182] * cs[15];
            RB[182] = RB[182] * cs[8];
            RF[183] = RF[183] * cs[8] * cs[15];
            RB[183] = RB[183] * cs[9];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15] + 2e0 * cs[13] + 2e0 * cs[14];
            RKLOW = exp(1.38440285e2 - 1.2e1 * logT - 3.00309643e3 * Tinv);
            PR = RKLOW * CTB / RF[184];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 9.8e-1 * exp(-T / 1.0966e3) + 2e-2 * exp(-T / 1.0966e3) +
                    exp(-6.8595e3 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[184] = RF[184] * PCOR;
            RB[184] = RB[184] * PCOR;
            RF[184] = RF[184] * cs[1] * cs[19];
            RB[184] = RB[184] * cs[20];
            RF[185] = RF[185] * cs[1] * cs[19];
            RB[185] = RB[185] * cs[9];
            RF[186] = RF[186] * cs[6] * cs[19];
            RB[186] = RB[186] * cs[3] * cs[20];
            RF[187] = RF[187] * cs[6] * cs[19];
            RB[187] = RB[187] * cs[4] * cs[12];
            RF[188] = RF[188] * cs[19];
            RB[188] = RB[188] * cs[10] * cs[20];
            CTB = CTOT + cs[0] + 5e0 * cs[5] + cs[9] + 5e-1 * cs[10] + cs[11] +
                  2e0 * cs[15];
            RKLOW = exp(8.93324137e1 - 6.66e0 * logT - 3.52251667e3 * Tinv);
            PR = RKLOW * CTB / RF[189];
            PCOR = PR / (1.0 + PR);
            PRLOG = log10(PR);
            FCENT = 0e0 * exp(-T / 1e3) + 1e0 * exp(-T / 1.31e3) +
                    exp(-4.8097e4 / T);
            FCLOG = log10(FCENT);
            XN = 0.75 - 1.27 * FCLOG;
            CPRLOG = PRLOG - (0.4 + 0.67 * FCLOG);
            FLOG = FCLOG / (1.0 + pow(CPRLOG / (XN - 0.14 * CPRLOG), 2.0));
            FC = pow(10.0, FLOG);
            PCOR = FC * PCOR;
            RF[189] = RF[189] * PCOR;
            RB[189] = RB[189] * PCOR;
            RF[189] = RF[189] * cs[1] * cs[20];
            RF[190] = RF[190] * cs[1] * cs[20];
            RB[190] = RB[190] * cs[8] * cs[14];
            RF[191] = RF[191] * cs[1] * cs[20];
            RB[191] = RB[191] * cs[0] * cs[19];
            RF[192] = RF[192] * cs[2] * cs[20];
            RB[192] = RB[192] * cs[1] * cs[8] * cs[17];
            RF[193] = RF[193] * cs[2] * cs[20];
            RF[194] = RF[194] * cs[2] * cs[20];
            RB[194] = RB[194] * cs[4] * cs[19];
            RF[195] = RF[195] * cs[4] * cs[20];
            RB[195] = RB[195] * cs[5] * cs[19];
            RF[196] = RF[196] * cs[6] * cs[20];
            RB[196] = RB[196] * cs[7] * cs[19];
            RF[197] = RF[197] * cs[8] * cs[20];
            RB[197] = RB[197] * cs[9] * cs[19];
            RF[198] = RF[198] * cs[1];
            RB[198] = RB[198] * cs[8];
            RF[199] = RF[199] * cs[1];
            RB[199] = RB[199] * cs[0] * cs[20];
            RF[200] = RF[200] * cs[2];
            RB[200] = RB[200] * cs[12];
            RF[201] = RF[201] * cs[4];
            RB[201] = RB[201] * cs[5] * cs[20];
            RF[202] = RF[202] * cs[3];
            RB[202] = RB[202] * cs[6] * cs[20];
            RF[203] = RF[203] * cs[6];
            RB[203] = RB[203] * cs[4] * cs[12];
            RF[204] = RF[204] * cs[8];
            RB[204] = RB[204] * cs[9] * cs[20];
            RB[205] = RB[205] * cs[8] * cs[19];
          } // end scope

          { // start scope
            double DEN;
            RF[56] = 0.0;
            RF[57] = 0.0;
            RF[142] = 0.0;
            RF[178] = 0.0;
            RB[193] = 0.0;
            RF[205] = 0.0;
            //    CH
            DEN = RF[33] + RF[34] + RF[35] + RF[36] + RF[37] + RF[38] + RF[39] +
                  RF[76] + RF[86] + RF[104] + RF[110] + RB[53] + RB[59];
            double A1_0 = (RB[33] + RB[36] + RB[38] + RB[56] + RB[76] +
                           RB[104] + RB[110]) /
                          DEN;
            double A1_2 = (RB[35] + RF[53]) / DEN;
            double A1_3 = (RF[59]) / DEN;
            double A1_4 = (RB[34] + RB[37] + RB[39]) / DEN;
            double A1_7 = (RB[86]) / DEN;
            //   CH2
            DEN = RF[47] + RF[48] + RF[49] + RF[50] + RF[51] + RF[52] + RF[53] +
                  RF[54] + RF[55] + RF[90] + RF[105] + RF[111] + RF[164] +
                  RB[35] + RB[58] + RB[66] + RB[67] + RB[68] + RB[79] +
                  RB[116] + RB[122] + RB[124] + RB[129] + RB[159];
            double A2_0 = (RB[47] + RB[48] + RB[51] + RB[52] + RB[54] + RB[55] +
                           RB[56] + RB[57] + RB[57] + RF[79] + RB[90] +
                           RB[105] + RF[116] + RF[129] + RF[159] + RB[164]) /
                          DEN;
            double A2_1 = (RF[35] + RB[53]) / DEN;
            double A2_3 = (RF[58] + RF[66] + RF[67] + RF[68]) / DEN;
            double A2_4 = (RB[49] + RB[50]) / DEN;
            double A2_6 = (RF[122] + RF[124]) / DEN;
            double A2_7 = (RB[111]) / DEN;
            //   CH2*
            DEN = RF[58] + RF[59] + RF[60] + RF[61] + RF[62] + RF[63] + RF[64] +
                  RF[65] + RF[66] + RF[67] + RF[68] + RF[69] + RF[91] +
                  RF[106] + RF[165] + RF[166] + RF[182] + RB[80] + RB[97] +
                  RB[107];
            double A3_0 = (RB[60] + RB[62] + RB[63] + RB[64] + RB[65] + RB[69] +
                           RF[80] + RB[91] + RB[106] + RF[107] + RB[166]) /
                          DEN;

            double A3_1 = (RB[59]) / DEN;
            double A3_2 = (RB[58] + RB[66] + RB[67] + RB[68]) / DEN;
            double A3_4 = (RB[61]) / DEN;
            double A3_5 = (RF[97]) / DEN;
            double A3_6 = (RB[165]) / DEN;
            double A3_8 = (RB[182]) / DEN;
            //   HCO
            DEN = RF[40] + RF[41] + RF[42] + RF[43] + RF[44] + RF[45] + RF[46] +
                  RF[87] + RF[88] + RF[119] + RF[163] + RF[188] + RB[34] +
                  RB[37] + RB[39] + RB[49] + RB[50] + RB[61] + RB[71] + RB[72] +
                  RB[73] + RB[74] + RB[75] + RB[89] + RB[139] + RB[148] +
                  RB[158];
            double A4_0 =
                (RB[40] + RB[41] + RB[42] + RB[43] + RB[44] + RB[45] + RB[46] +
                 RF[71] + RF[72] + RF[73] + RF[74] + RF[75] + RB[87] + RB[88] +
                 RF[89] + RB[142] + RF[158] + RB[178] + RB[188] + RF[193]) /
                DEN;
            double A4_1 = (RF[34] + RF[37] + RF[39]) / DEN;
            double A4_2 = (RF[49] + RF[50]) / DEN;
            double A4_3 = (RF[61]) / DEN;
            double A4_7 = (RB[119] + RF[139]) / DEN;
            double A4_8 = (RB[163]) / DEN;
            double A4_9 = (RF[148]) / DEN;
            //   CH3O
            DEN = RF[95] + RF[96] + RF[97] + RF[98] + RF[99] + RF[100] +
                  RB[70] + RB[81] + RB[84];
            double A5_0 = (RF[70] + RF[81] + RF[84] + RB[95] + RB[96] + RB[98] +
                           RB[99] + RB[100]) /
                          DEN;
            double A5_3 = (RB[97]) / DEN;
            //   H2CC
            DEN = RF[121] + RF[122] + RF[123] + RF[124] + RB[113] + RB[133] +
                  RB[154] + RB[165] + RB[185];
            double A6_0 =
                (RF[113] + RB[121] + RB[123] + RF[154] + RF[185]) / DEN;
            double A6_2 = (RB[122] + RB[124]) / DEN;
            double A6_3 = (RF[165]) / DEN;
            double A6_7 = (RF[133]) / DEN;
            //   C2H3
            DEN = RF[114] + RF[131] + RF[132] + RF[133] + RF[134] + RF[135] +
                  RF[136] + RF[137] + RF[138] + RF[139] + RF[140] + RF[141] +
                  RF[143] + RF[144] + RF[145] + RB[86] + RB[111] + RB[119] +
                  RB[156] + RB[157] + RB[160] + RB[161] + RB[167] + RB[187];
            double A7_0 = (RB[114] + RB[131] + RB[132] + RB[134] + RB[135] +
                           RB[136] + RB[137] + RB[141] + RB[142] + RB[143] +
                           RB[144] + RB[145] + RF[156] + RF[157] + RF[160] +
                           RF[161] + RF[167] + RF[187] + RB[205]) /
                          DEN;
            double A7_1 = (RF[86]) / DEN;
            double A7_2 = (RF[111]) / DEN;
            double A7_4 = (RF[119] + RB[139]) / DEN;
            double A7_6 = (RB[133]) / DEN;
            double A7_9 = (RB[138] + RB[140]) / DEN;
            //   C2H5
            DEN = RF[169] + RF[170] + RF[171] + RF[172] + RF[173] + RF[174] +
                  RF[175] + RF[176] + RF[177] + RB[93] + RB[155] + RB[163] +
                  RB[179] + RB[180] + RB[181] + RB[182] + RB[183] + RB[198] +
                  RB[200] + RB[203];
            double A8_0 =
                (RF[93] + RF[155] + RB[169] + RB[170] + RB[171] + RB[172] +
                 RB[173] + RB[174] + RB[175] + RB[176] + RB[177] + RB[178] +
                 RF[179] + RF[180] + RF[181] + RF[183] + RF[193] + RB[205]) /
                DEN;
            double A8_3 = (RF[182]) / DEN;
            double A8_4 = (RF[163]) / DEN;
            double A8_10 = (RF[198] + RF[200] + RF[203]) / DEN;
            //   CH2CHO
            DEN = RF[146] + RF[147] + RF[148] + RF[149] + RF[150] + RF[151] +
                  RF[152] + RF[153] + RB[125] + RB[138] + RB[140];
            double A9_0 = (RF[125] + RB[146] + RB[147] + RB[149] + RB[150] +
                           RB[151] + RB[152] + RB[153]) /
                          DEN;
            double A9_4 = (RB[148]) / DEN;
            double A9_7 = (RF[138] + RF[140]) / DEN;
            //   nC3H7
            DEN = RF[198] + RF[199] + RF[200] + RF[201] + RF[202] + RF[203] +
                  RF[204] + RB[168] + RB[189];
            double A10_0 =
                (RF[168] + RF[189] + RB[199] + RB[201] + RB[202] + RB[204]) /
                DEN;
            double A10_8 = (RB[198] + RB[200] + RB[203]) / DEN;
            //

            A3_0 = A3_0 + A3_5 * A5_0;
            DEN = 1.0 - A3_5 * A5_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A3_2 = A3_2 / DEN;
            A3_1 = A3_1 / DEN;
            A3_6 = A3_6 / DEN;
            A3_8 = A3_8 / DEN;
            A8_0 = A8_0 + A8_10 * A10_0;
            DEN = 1.0 - A8_10 * A10_8;
            A8_0 = A8_0 / DEN;
            A8_3 = A8_3 / DEN;
            A8_4 = A8_4 / DEN;
            A4_0 = A4_0 + A4_9 * A9_0;
            A4_7 = A4_7 + A4_9 * A9_7;
            DEN = 1.0 - A4_9 * A9_4;
            A4_0 = A4_0 / DEN;
            A4_3 = A4_3 / DEN;
            A4_7 = A4_7 / DEN;
            A4_2 = A4_2 / DEN;
            A4_1 = A4_1 / DEN;
            A4_8 = A4_8 / DEN;
            A7_0 = A7_0 + A7_9 * A9_0;
            A7_4 = A7_4 + A7_9 * A9_4;
            DEN = 1.0 - A7_9 * A9_7;
            A7_0 = A7_0 / DEN;
            A7_4 = A7_4 / DEN;
            A7_2 = A7_2 / DEN;
            A7_1 = A7_1 / DEN;
            A7_6 = A7_6 / DEN;
            A3_0 = A3_0 + A3_8 * A8_0;
            A3_4 = A3_4 + A3_8 * A8_4;
            DEN = 1.0 - A3_8 * A8_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A3_2 = A3_2 / DEN;
            A3_1 = A3_1 / DEN;
            A3_6 = A3_6 / DEN;
            A4_0 = A4_0 + A4_8 * A8_0;
            A4_3 = A4_3 + A4_8 * A8_3;
            DEN = 1.0 - A4_8 * A8_4;
            A4_0 = A4_0 / DEN;
            A4_3 = A4_3 / DEN;
            A4_7 = A4_7 / DEN;
            A4_2 = A4_2 / DEN;
            A4_1 = A4_1 / DEN;
            A3_0 = A3_0 + A3_6 * A6_0;
            double A3_7 = A3_6 * A6_7;
            A3_2 = A3_2 + A3_6 * A6_2;
            DEN = 1.0 - A3_6 * A6_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A3_7 = A3_7 / DEN;
            A3_2 = A3_2 / DEN;
            A3_1 = A3_1 / DEN;
            A7_0 = A7_0 + A7_6 * A6_0;
            double A7_3 = A7_6 * A6_3;
            A7_2 = A7_2 + A7_6 * A6_2;
            DEN = 1.0 - A7_6 * A6_7;
            A7_0 = A7_0 / DEN;
            A7_3 = A7_3 / DEN;
            A7_4 = A7_4 / DEN;
            A7_2 = A7_2 / DEN;
            A7_1 = A7_1 / DEN;
            A2_0 = A2_0 + A2_6 * A6_0;
            A2_3 = A2_3 + A2_6 * A6_3;
            A2_7 = A2_7 + A2_6 * A6_7;
            DEN = 1.0 - A2_6 * A6_2;
            A2_0 = A2_0 / DEN;
            A2_3 = A2_3 / DEN;
            A2_4 = A2_4 / DEN;
            A2_7 = A2_7 / DEN;
            A2_1 = A2_1 / DEN;
            A3_0 = A3_0 + A3_1 * A1_0;
            A3_4 = A3_4 + A3_1 * A1_4;
            A3_7 = A3_7 + A3_1 * A1_7;
            A3_2 = A3_2 + A3_1 * A1_2;
            DEN = 1.0 - A3_1 * A1_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A3_7 = A3_7 / DEN;
            A3_2 = A3_2 / DEN;
            A4_0 = A4_0 + A4_1 * A1_0;
            A4_3 = A4_3 + A4_1 * A1_3;
            A4_7 = A4_7 + A4_1 * A1_7;
            A4_2 = A4_2 + A4_1 * A1_2;
            DEN = 1.0 - A4_1 * A1_4;
            A4_0 = A4_0 / DEN;
            A4_3 = A4_3 / DEN;
            A4_7 = A4_7 / DEN;
            A4_2 = A4_2 / DEN;
            A7_0 = A7_0 + A7_1 * A1_0;
            A7_3 = A7_3 + A7_1 * A1_3;
            A7_4 = A7_4 + A7_1 * A1_4;
            A7_2 = A7_2 + A7_1 * A1_2;
            DEN = 1.0 - A7_1 * A1_7;
            A7_0 = A7_0 / DEN;
            A7_3 = A7_3 / DEN;
            A7_4 = A7_4 / DEN;
            A7_2 = A7_2 / DEN;
            A2_0 = A2_0 + A2_1 * A1_0;
            A2_3 = A2_3 + A2_1 * A1_3;
            A2_4 = A2_4 + A2_1 * A1_4;
            A2_7 = A2_7 + A2_1 * A1_7;
            DEN = 1.0 - A2_1 * A1_2;
            A2_0 = A2_0 / DEN;
            A2_3 = A2_3 / DEN;
            A2_4 = A2_4 / DEN;
            A2_7 = A2_7 / DEN;
            A3_0 = A3_0 + A3_2 * A2_0;
            A3_4 = A3_4 + A3_2 * A2_4;
            A3_7 = A3_7 + A3_2 * A2_7;
            DEN = 1.0 - A3_2 * A2_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A3_7 = A3_7 / DEN;
            A4_0 = A4_0 + A4_2 * A2_0;
            A4_3 = A4_3 + A4_2 * A2_3;
            A4_7 = A4_7 + A4_2 * A2_7;
            DEN = 1.0 - A4_2 * A2_4;
            A4_0 = A4_0 / DEN;
            A4_3 = A4_3 / DEN;
            A4_7 = A4_7 / DEN;
            A7_0 = A7_0 + A7_2 * A2_0;
            A7_3 = A7_3 + A7_2 * A2_3;
            A7_4 = A7_4 + A7_2 * A2_4;
            DEN = 1.0 - A7_2 * A2_7;
            A7_0 = A7_0 / DEN;
            A7_3 = A7_3 / DEN;
            A7_4 = A7_4 / DEN;
            A3_0 = A3_0 + A3_7 * A7_0;
            A3_4 = A3_4 + A3_7 * A7_4;
            DEN = 1.0 - A3_7 * A7_3;
            A3_0 = A3_0 / DEN;
            A3_4 = A3_4 / DEN;
            A4_0 = A4_0 + A4_7 * A7_0;
            A4_3 = A4_3 + A4_7 * A7_3;
            DEN = 1.0 - A4_7 * A7_4;
            A4_0 = A4_0 / DEN;
            A4_3 = A4_3 / DEN;
            A3_0 = A3_0 + A3_4 * A4_0;
            DEN = 1.0 - A3_4 * A4_3;
            A3_0 = A3_0 / DEN;

            double XQ[10];
            XQ[2] = A3_0;
            XQ[3] = A4_0 + A4_3 * XQ[2];
            XQ[6] = A7_0 + A7_3 * XQ[2] + A7_4 * XQ[3];
            XQ[1] = A2_0 + A2_3 * XQ[2] + A2_4 * XQ[3] + A2_7 * XQ[6];
            XQ[0] = A1_0 + A1_3 * XQ[2] + A1_4 * XQ[3] + A1_7 * XQ[6] +
                    A1_2 * XQ[1];
            XQ[5] = A6_0 + A6_3 * XQ[2] + A6_7 * XQ[6] + A6_2 * XQ[1];
            XQ[7] = A8_0 + A8_3 * XQ[2] + A8_4 * XQ[3];
            XQ[8] = A9_0 + A9_4 * XQ[3] + A9_7 * XQ[6];
            XQ[9] = A10_0 + A10_8 * XQ[7];
            XQ[4] = A5_0 + A5_3 * XQ[2];

            RF[33] = RF[33] * XQ[0];
            RF[34] = RF[34] * XQ[0];
            RB[34] = RB[34] * XQ[3];
            RF[35] = RF[35] * XQ[0];
            RB[35] = RB[35] * XQ[1];
            RF[36] = RF[36] * XQ[0];
            RF[37] = RF[37] * XQ[0];
            RB[37] = RB[37] * XQ[3];
            RF[38] = RF[38] * XQ[0];
            RF[39] = RF[39] * XQ[0];
            RB[39] = RB[39] * XQ[3];
            RF[40] = RF[40] * XQ[3];
            RF[41] = RF[41] * XQ[3];
            RF[42] = RF[42] * XQ[3];
            RF[43] = RF[43] * XQ[3];
            RF[44] = RF[44] * XQ[3];
            RF[45] = RF[45] * XQ[3];
            RF[46] = RF[46] * XQ[3];
            RF[47] = RF[47] * XQ[1];
            RF[48] = RF[48] * XQ[1];
            RF[49] = RF[49] * XQ[1];
            RB[49] = RB[49] * XQ[3];
            RF[50] = RF[50] * XQ[1];
            RB[50] = RB[50] * XQ[3];
            RF[51] = RF[51] * XQ[1];
            RF[52] = RF[52] * XQ[1];
            RF[53] = RF[53] * XQ[1];
            RB[53] = RB[53] * XQ[0];
            RF[54] = RF[54] * XQ[1];
            RF[55] = RF[55] * XQ[1];
            RF[58] = RF[58] * XQ[2];
            RB[58] = RB[58] * XQ[1];
            RF[59] = RF[59] * XQ[2];
            RB[59] = RB[59] * XQ[0];
            RF[60] = RF[60] * XQ[2];
            RF[61] = RF[61] * XQ[2];
            RB[61] = RB[61] * XQ[3];
            RF[62] = RF[62] * XQ[2];
            RF[63] = RF[63] * XQ[2];
            RF[64] = RF[64] * XQ[2];
            RF[65] = RF[65] * XQ[2];
            RF[66] = RF[66] * XQ[2];
            RB[66] = RB[66] * XQ[1];
            RF[67] = RF[67] * XQ[2];
            RB[67] = RB[67] * XQ[1];
            RF[68] = RF[68] * XQ[2];
            RB[68] = RB[68] * XQ[1];
            RF[69] = RF[69] * XQ[2];
            RB[70] = RB[70] * XQ[4];
            RB[71] = RB[71] * XQ[3];
            RB[72] = RB[72] * XQ[3];
            RB[73] = RB[73] * XQ[3];
            RB[74] = RB[74] * XQ[3];
            RB[75] = RB[75] * XQ[3];
            RF[76] = RF[76] * XQ[0];
            RB[79] = RB[79] * XQ[1];
            RB[80] = RB[80] * XQ[2];
            RB[81] = RB[81] * XQ[4];
            RB[84] = RB[84] * XQ[4];
            RF[86] = RF[86] * XQ[0];
            RB[86] = RB[86] * XQ[6];
            RF[87] = RF[87] * XQ[3];
            RF[88] = RF[88] * XQ[3];
            RB[89] = RB[89] * XQ[3];
            RF[90] = RF[90] * XQ[1];
            RF[91] = RF[91] * XQ[2];
            RB[93] = RB[93] * XQ[7];
            RF[95] = RF[95] * XQ[4];
            RF[96] = RF[96] * XQ[4];
            RF[97] = RF[97] * XQ[4];
            RB[97] = RB[97] * XQ[2];
            RF[98] = RF[98] * XQ[4];
            RF[99] = RF[99] * XQ[4];
            RF[100] = RF[100] * XQ[4];
            RF[104] = RF[104] * XQ[0];
            RF[105] = RF[105] * XQ[1];
            RF[106] = RF[106] * XQ[2];
            RB[107] = RB[107] * XQ[2];
            RF[110] = RF[110] * XQ[0];
            RF[111] = RF[111] * XQ[1];
            RB[111] = RB[111] * XQ[6];
            RB[113] = RB[113] * XQ[5];
            RF[114] = RF[114] * XQ[6];
            RB[116] = RB[116] * XQ[1];
            RF[119] = RF[119] * XQ[3];
            RB[119] = RB[119] * XQ[6];
            RF[121] = RF[121] * XQ[5];
            RF[122] = RF[122] * XQ[5];
            RB[122] = RB[122] * XQ[1];
            RF[123] = RF[123] * XQ[5];
            RF[124] = RF[124] * XQ[5];
            RB[124] = RB[124] * XQ[1];
            RB[125] = RB[125] * XQ[8];
            RB[129] = RB[129] * XQ[1];
            RF[131] = RF[131] * XQ[6];
            RF[132] = RF[132] * XQ[6];
            RF[133] = RF[133] * XQ[6];
            RB[133] = RB[133] * XQ[5];
            RF[134] = RF[134] * XQ[6];
            RF[135] = RF[135] * XQ[6];
            RF[136] = RF[136] * XQ[6];
            RF[137] = RF[137] * XQ[6];
            RF[138] = RF[138] * XQ[6];
            RB[138] = RB[138] * XQ[8];
            RF[139] = RF[139] * XQ[6];
            RB[139] = RB[139] * XQ[3];
            RF[140] = RF[140] * XQ[6];
            RB[140] = RB[140] * XQ[8];
            RF[141] = RF[141] * XQ[6];
            RF[143] = RF[143] * XQ[6];
            RF[144] = RF[144] * XQ[6];
            RF[145] = RF[145] * XQ[6];
            RF[146] = RF[146] * XQ[8];
            RF[147] = RF[147] * XQ[8];
            RF[148] = RF[148] * XQ[8];
            RB[148] = RB[148] * XQ[3];
            RF[149] = RF[149] * XQ[8];
            RF[150] = RF[150] * XQ[8];
            RF[151] = RF[151] * XQ[8];
            RF[152] = RF[152] * XQ[8];
            RF[153] = RF[153] * XQ[8];
            RB[154] = RB[154] * XQ[5];
            RB[155] = RB[155] * XQ[7];
            RB[156] = RB[156] * XQ[6];
            RB[157] = RB[157] * XQ[6];
            RB[158] = RB[158] * XQ[3];
            RB[159] = RB[159] * XQ[1];
            RB[160] = RB[160] * XQ[6];
            RB[161] = RB[161] * XQ[6];
            RF[163] = RF[163] * XQ[3];
            RB[163] = RB[163] * XQ[7];
            RF[164] = RF[164] * XQ[1];
            RF[165] = RF[165] * XQ[2];
            RB[165] = RB[165] * XQ[5];
            RF[166] = RF[166] * XQ[2];
            RB[167] = RB[167] * XQ[6];
            RB[168] = RB[168] * XQ[9];
            RF[169] = RF[169] * XQ[7];
            RF[170] = RF[170] * XQ[7];
            RF[171] = RF[171] * XQ[7];
            RF[172] = RF[172] * XQ[7];
            RF[173] = RF[173] * XQ[7];
            RF[174] = RF[174] * XQ[7];
            RF[175] = RF[175] * XQ[7];
            RF[176] = RF[176] * XQ[7];
            RF[177] = RF[177] * XQ[7];
            RB[179] = RB[179] * XQ[7];
            RB[180] = RB[180] * XQ[7];
            RB[181] = RB[181] * XQ[7];
            RF[182] = RF[182] * XQ[2];
            RB[182] = RB[182] * XQ[7];
            RB[183] = RB[183] * XQ[7];
            RB[185] = RB[185] * XQ[5];
            RB[187] = RB[187] * XQ[6];
            RF[188] = RF[188] * XQ[3];
            RB[189] = RB[189] * XQ[9];
            RF[198] = RF[198] * XQ[9];
            RB[198] = RB[198] * XQ[7];
            RF[199] = RF[199] * XQ[9];
            RF[200] = RF[200] * XQ[9];
            RB[200] = RB[200] * XQ[7];
            RF[201] = RF[201] * XQ[9];
            RF[202] = RF[202] * XQ[9];
            RF[203] = RF[203] * XQ[9];
            RB[203] = RB[203] * XQ[7];
            RF[204] = RF[204] * XQ[9];
          } // end scope

          { // start scope
            double ROP;
            ROP = RF[0] - RB[0];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[2] = dYdt[2] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            ROP = RF[1] - RB[1];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            ROP = RF[2] - RB[2];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            ROP = RF[3] - RB[3];
            dYdt[2] = dYdt[2] + ROP;
            dYdt[4] = dYdt[4] - ROP - ROP;
            dYdt[5] = dYdt[5] + ROP;
            ROP = RF[4] - RB[4];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP - ROP;
            ROP = RF[5] - RB[5];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP - ROP;
            ROP = RF[6] - RB[6];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP - ROP;
            ROP = RF[7] - RB[7];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP - ROP;
            ROP = RF[8] - RB[8];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            ROP = RF[9] - RB[9];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            ROP = RF[10] - RB[10];
            dYdt[2] = dYdt[2] - ROP - ROP;
            dYdt[3] = dYdt[3] + ROP;
            ROP = RF[11] - RB[11];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            ROP = RF[12] - RB[12];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            ROP = RF[13] - RB[13];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            ROP = RF[14] - RB[14];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            ROP = RF[15] - RB[15];
            dYdt[4] = dYdt[4] - ROP - ROP;
            dYdt[7] = dYdt[7] + ROP;
            ROP = RF[16] - RB[16];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[2] = dYdt[2] + ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[17] - RB[17];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[18] - RB[18];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[4] = dYdt[4] + ROP + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[19] - RB[19];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[3] = dYdt[3] + ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[20] - RB[20];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[21] - RB[21];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP - ROP;
            dYdt[7] = dYdt[7] + ROP;
            ROP = RF[22] - RB[22];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP - ROP;
            dYdt[7] = dYdt[7] + ROP;
            ROP = RF[23] - RB[23];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            ROP = RF[24] - RB[24];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            ROP = RF[25] - RB[25];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            ROP = RF[26] - RB[26];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            ROP = RF[27] - RB[27];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            ROP = RF[28] - RB[28];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[29] - RB[29];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[10] = dYdt[10] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[30] - RB[30];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[10] = dYdt[10] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[31] - RB[31];
            dYdt[2] = dYdt[2] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[10] = dYdt[10] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[32] - RB[32];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[10] = dYdt[10] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[33] - RB[33];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[34] - RB[34];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            ROP = RF[35] - RB[35];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[1] = dYdt[1] + ROP;
            ROP = RF[36] - RB[36];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[5] = dYdt[5] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[37] - RB[37];
            dYdt[2] = dYdt[2] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            ROP = RF[38] - RB[38];
            dYdt[10] = dYdt[10] - ROP;
            dYdt[16] = dYdt[16] + ROP;
            ROP = RF[39] - RB[39];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[11] = dYdt[11] - ROP;
            ROP = RF[40] - RB[40];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[41] - RB[41];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[42] - RB[42];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[43] - RB[43];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[44] - RB[44];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[45] - RB[45];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[46] - RB[46];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[47] - RB[47];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[48] - RB[48];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[49] - RB[49];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            ROP = RF[50] - RB[50];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            ROP = RF[51] - RB[51];
            dYdt[1] = dYdt[1] + ROP + ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[52] - RB[52];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[53] - RB[53];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            ROP = RF[54] - RB[54];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[55] - RB[55];
            dYdt[10] = dYdt[10] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[56] - RB[56];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[57] - RB[57];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[58] - RB[58];
            ROP = RF[59] - RB[59];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            ROP = RF[60] - RB[60];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[61] - RB[61];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            ROP = RF[62] - RB[62];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[63] - RB[63];
            dYdt[0] = dYdt[0] - ROP;
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[64] - RB[64];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[65] - RB[65];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[66] - RB[66];
            ROP = RF[67] - RB[67];
            ROP = RF[68] - RB[68];
            ROP = RF[69] - RB[69];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[11] = dYdt[11] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[70] - RB[70];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[71] - RB[71];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[72] - RB[72];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[73] - RB[73];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[74] - RB[74];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[75] - RB[75];
            dYdt[6] = dYdt[6] - ROP;
            dYdt[7] = dYdt[7] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[76] - RB[76];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[77] - RB[77];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            ROP = RF[78] - RB[78];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[79] - RB[79];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            ROP = RF[80] - RB[80];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            ROP = RF[81] - RB[81];
            dYdt[2] = dYdt[2] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            ROP = RF[82] - RB[82];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[83] - RB[83];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            ROP = RF[84] - RB[84];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            ROP = RF[85] - RB[85];
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            ROP = RF[86] - RB[86];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            ROP = RF[87] - RB[87];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[88] - RB[88];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[18] = dYdt[18] + ROP;
            ROP = RF[89] - RB[89];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[12] = dYdt[12] - ROP;
            ROP = RF[90] - RB[90];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[91] - RB[91];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[92] - RB[92];
            dYdt[8] = dYdt[8] - ROP - ROP;
            dYdt[15] = dYdt[15] + ROP;
            ROP = RF[93] - RB[93];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] - ROP - ROP;
            ROP = RF[94] - RB[94];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[14] = dYdt[14] + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[95] - RB[95];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[96] - RB[96];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[97] - RB[97];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            ROP = RF[98] - RB[98];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[99] - RB[99];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[100] - RB[100];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[101] - RB[101];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[9] = dYdt[9] - ROP;
            ROP = RF[102] - RB[102];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[9] = dYdt[9] - ROP;
            ROP = RF[103] - RB[103];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[9] = dYdt[9] - ROP;
            ROP = RF[104] - RB[104];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[9] = dYdt[9] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[105] - RB[105];
            dYdt[8] = dYdt[8] + ROP + ROP;
            dYdt[9] = dYdt[9] - ROP;
            ROP = RF[106] - RB[106];
            dYdt[8] = dYdt[8] + ROP + ROP;
            dYdt[9] = dYdt[9] - ROP;
            ROP = RF[107] - RB[107];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[108] - RB[108];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] + ROP + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[109] - RB[109];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[10] = dYdt[10] + ROP + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[110] - RB[110];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[111] - RB[111];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[16] = dYdt[16] - ROP;
            ROP = RF[112] - RB[112];
            dYdt[10] = dYdt[10] + ROP + ROP;
            dYdt[13] = dYdt[13] + ROP;
            dYdt[16] = dYdt[16] - ROP - ROP;
            ROP = RF[113] - RB[113];
            dYdt[13] = dYdt[13] - ROP;
            ROP = RF[114] - RB[114];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[115] - RB[115];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[13] = dYdt[13] - ROP;
            dYdt[16] = dYdt[16] + ROP;
            ROP = RF[116] - RB[116];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[13] = dYdt[13] - ROP;
            ROP = RF[117] - RB[117];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[13] = dYdt[13] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[118] - RB[118];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[13] = dYdt[13] - ROP;
            ROP = RF[119] - RB[119];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[13] = dYdt[13] - ROP;
            ROP = RF[120] - RB[120];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[13] = dYdt[13] - ROP;
            dYdt[19] = dYdt[19] + ROP;
            ROP = RF[121] - RB[121];
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[122] - RB[122];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[123] - RB[123];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[4] = dYdt[4] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[124] - RB[124];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            ROP = RF[125] - RB[125];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[126] - RB[126];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[16] = dYdt[16] + ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[127] - RB[127];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[128] - RB[128];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[16] = dYdt[16] + ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[129] - RB[129];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[11] = dYdt[11] + ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[130] - RB[130];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[16] = dYdt[16] + ROP;
            dYdt[17] = dYdt[17] - ROP;
            ROP = RF[131] - RB[131];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[132] - RB[132];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[133] - RB[133];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            ROP = RF[134] - RB[134];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[135] - RB[135];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[136] - RB[136];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[137] - RB[137];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[138] - RB[138];
            dYdt[2] = dYdt[2] + ROP;
            dYdt[3] = dYdt[3] - ROP;
            ROP = RF[139] - RB[139];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[140] - RB[140];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            ROP = RF[141] - RB[141];
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[142] - RB[142];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[143] - RB[143];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[13] = dYdt[13] + ROP;
            ROP = RF[144] - RB[144];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[145] - RB[145];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[8] = dYdt[8] - ROP;
            dYdt[19] = dYdt[19] + ROP;
            ROP = RF[146] - RB[146];
            dYdt[8] = dYdt[8] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            ROP = RF[147] - RB[147];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[18] = dYdt[18] + ROP;
            ROP = RF[148] - RB[148];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[149] - RB[149];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[150] - RB[150];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[151] - RB[151];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[152] - RB[152];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[17] = dYdt[17] + ROP;
            ROP = RF[153] - RB[153];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[10] = dYdt[10] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[154] - RB[154];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[155] - RB[155];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[156] - RB[156];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[157] - RB[157];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[158] - RB[158];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[159] - RB[159];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[160] - RB[160];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[161] - RB[161];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[162] - RB[162];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[14] = dYdt[14] - ROP;
            dYdt[18] = dYdt[18] + ROP;
            ROP = RF[163] - RB[163];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[164] - RB[164];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            dYdt[19] = dYdt[19] + ROP;
            ROP = RF[165] - RB[165];
            dYdt[9] = dYdt[9] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[166] - RB[166];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            dYdt[19] = dYdt[19] + ROP;
            ROP = RF[167] - RB[167];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[168] - RB[168];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[14] = dYdt[14] - ROP;
            ROP = RF[169] - RB[169];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[15] = dYdt[15] + ROP;
            ROP = RF[170] - RB[170];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[171] - RB[171];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[172] - RB[172];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[18] = dYdt[18] + ROP;
            ROP = RF[173] - RB[173];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[174] - RB[174];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[15] = dYdt[15] + ROP;
            ROP = RF[175] - RB[175];
            dYdt[6] = dYdt[6] - ROP;
            dYdt[7] = dYdt[7] + ROP;
            dYdt[14] = dYdt[14] + ROP;
            ROP = RF[176] - RB[176];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[177] - RB[177];
            dYdt[6] = dYdt[6] + ROP;
            dYdt[7] = dYdt[7] - ROP;
            dYdt[15] = dYdt[15] + ROP;
            ROP = RF[178] - RB[178];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[15] = dYdt[15] + ROP;
            ROP = RF[179] - RB[179];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[15] = dYdt[15] - ROP;
            ROP = RF[180] - RB[180];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[15] = dYdt[15] - ROP;
            ROP = RF[181] - RB[181];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[15] = dYdt[15] - ROP;
            ROP = RF[182] - RB[182];
            dYdt[8] = dYdt[8] + ROP;
            dYdt[15] = dYdt[15] - ROP;
            ROP = RF[183] - RB[183];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[15] = dYdt[15] - ROP;
            ROP = RF[184] - RB[184];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[19] = dYdt[19] - ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[185] - RB[185];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[19] = dYdt[19] - ROP;
            ROP = RF[186] - RB[186];
            dYdt[3] = dYdt[3] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[19] = dYdt[19] - ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[187] - RB[187];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            dYdt[19] = dYdt[19] - ROP;
            ROP = RF[188] - RB[188];
            dYdt[10] = dYdt[10] + ROP;
            dYdt[19] = dYdt[19] - ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[189] - RB[189];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[190] - RB[190];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[14] = dYdt[14] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[191] - RB[191];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[19] = dYdt[19] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[192] - RB[192];
            dYdt[1] = dYdt[1] + ROP;
            dYdt[2] = dYdt[2] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            dYdt[17] = dYdt[17] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[193] - RB[193];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[194] - RB[194];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[4] = dYdt[4] + ROP;
            dYdt[19] = dYdt[19] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[195] - RB[195];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[19] = dYdt[19] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[196] - RB[196];
            dYdt[6] = dYdt[6] - ROP;
            dYdt[7] = dYdt[7] + ROP;
            dYdt[19] = dYdt[19] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[197] - RB[197];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[19] = dYdt[19] + ROP;
            dYdt[20] = dYdt[20] - ROP;
            ROP = RF[198] - RB[198];
            dYdt[1] = dYdt[1] - ROP;
            dYdt[8] = dYdt[8] + ROP;
            ROP = RF[199] - RB[199];
            dYdt[0] = dYdt[0] + ROP;
            dYdt[1] = dYdt[1] - ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[200] - RB[200];
            dYdt[2] = dYdt[2] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[201] - RB[201];
            dYdt[4] = dYdt[4] - ROP;
            dYdt[5] = dYdt[5] + ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[202] - RB[202];
            dYdt[3] = dYdt[3] - ROP;
            dYdt[6] = dYdt[6] + ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[203] - RB[203];
            dYdt[4] = dYdt[4] + ROP;
            dYdt[6] = dYdt[6] - ROP;
            dYdt[12] = dYdt[12] + ROP;
            ROP = RF[204] - RB[204];
            dYdt[8] = dYdt[8] - ROP;
            dYdt[9] = dYdt[9] + ROP;
            dYdt[20] = dYdt[20] + ROP;
            ROP = RF[205] - RB[205];
            dYdt[8] = dYdt[8] + ROP;
            dYdt[19] = dYdt[19] + ROP;
          } // end scope

          // convert from mol/cm^3 to kg/m^3
          dYdt[0] *= th.MW(0) * 1000.0;
          dYdt[1] *= th.MW(1) * 1000.0;
          dYdt[2] *= th.MW(2) * 1000.0;
          dYdt[3] *= th.MW(3) * 1000.0;
          dYdt[4] *= th.MW(4) * 1000.0;
          dYdt[5] *= th.MW(5) * 1000.0;
          dYdt[6] *= th.MW(6) * 1000.0;
          dYdt[7] *= th.MW(7) * 1000.0;
          dYdt[8] *= th.MW(8) * 1000.0;
          dYdt[9] *= th.MW(9) * 1000.0;
          dYdt[10] *= th.MW(10) * 1000.0;
          dYdt[11] *= th.MW(11) * 1000.0;
          dYdt[12] *= th.MW(12) * 1000.0;
          dYdt[13] *= th.MW(13) * 1000.0;
          dYdt[14] *= th.MW(14) * 1000.0;
          dYdt[15] *= th.MW(15) * 1000.0;
          dYdt[16] *= th.MW(16) * 1000.0;
          dYdt[17] *= th.MW(17) * 1000.0;
          dYdt[18] *= th.MW(18) * 1000.0;
          dYdt[19] *= th.MW(19) * 1000.0;
          dYdt[20] *= th.MW(20) * 1000.0;

          for (int n = 0; n < 21; n++) {
            if (Kokkos::isfinite(dYdt[n]) == false) {
              dYdt[n] = 0.0;
            }
          }
          dTdt = 0.0;
          for (int n = 0; n < 21; n++) {
            dTdt -= hi[n] * dYdt[n];
            Y[n] += dYdt[n] / rho * tSub;
          }
          dTdt /= cp * rho;
          T += dTdt * tSub;
        } // End of chem sub step for loop

        // Compute d(rhoYi)/dt based on where we end up
        // Add source terms to RHS
        for (int n = 0; n < 21; n++) {
          b.dQ(i, j, k, 5 + n) += (Y[n] * rho - b.Q(i, j, k, 5 + n)) / dt;
        }

        // Store dTdt and dYdt (for implicit chem integration)
        for (int n = 0; n < 21; n++) {
          b.omega(i, j, k, n + 1) = dYdt[n] / b.Q(i, j, k, 0);
        }
        b.omega(i, j, k, 0) = 0;
      });
}

// ========================================================== //
// Y(  0) = H2
// Y(  1) = H
// Y(  2) = O
// Y(  3) = O2
// Y(  4) = OH
// Y(  5) = H2O
// Y(  6) = HO2
// Y(  7) = H2O2
// Y(  8) = C
// Y(  9) = CH
// Y( 10) = CH2
// Y( 11) = CH2(S)
// Y( 12) = CH3
// Y( 13) = CH4
// Y( 14) = CO
// Y( 15) = CO2
// Y( 16) = HCO
// Y( 17) = CH2O
// Y( 18) = CH2OH
// Y( 19) = CH3O
// Y( 20) = CH3OH
// Y( 21) = C2H
// Y( 22) = C2H2
// Y( 23) = C2H3
// Y( 24) = C2H4
// Y( 25) = C2H5
// Y( 26) = C2H6
// Y( 27) = HCCO
// Y( 28) = CH2CO
// Y( 29) = HCCOH
// Y( 30) = N
// Y( 31) = NH
// Y( 32) = NH2
// Y( 33) = NH3
// Y( 34) = NNH
// Y( 35) = NO
// Y( 36) = NO2
// Y( 37) = N2O
// Y( 38) = HNO
// Y( 39) = CN
// Y( 40) = HCN
// Y( 41) = H2CN
// Y( 42) = HCNN
// Y( 43) = HCNO
// Y( 44) = HOCN
// Y( 45) = HNCO
// Y( 46) = NCO
// Y( 47) = N2
// Y( 48) = AR
// Y( 49) = C3H7
// Y( 50) = C3H8
// Y( 51) = CH2CHO
// Y( 52) = CH3CHO

// 325 reactions.
// ========================================================== //

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>

void chem_GRI30(block_ b, thtrdat_ th, int face/*=0*/, int indxI/*=0*/, int indxJ/*=0*/, int indxK/*=0*/) {

// --------------------------------------------------------------|
// cc range
// --------------------------------------------------------------|
  MDRange3 range = get_range3(b, face, indxI, indxJ, indxK);

  Kokkos::parallel_for("Compute chemical source terms",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double& T = b.q(i,j,k,4);
  double& rho = b.Q(i,j,k,0);
  double Y[53];
  const double logT = log(T);
  const double prefRuT = 101325.0/(th.Ru*T);

  // Compute nth species Y
  Y[52] = 1.0;
  for (int n=0; n<52; n++)
  {
    Y[n] = b.q(i,j,k,5+n);
    Y[52] -= Y[n];
  }
  Y[52] = fmax(0.0,Y[52]);

  // Conecntrations
  double cs[53];
  for (int n=0; n<=52; n++)
  {
    cs[n] = rho*Y[n]/th.MW(n);
  }

  // ----------------------------------------------------------- >
  // Chaperon efficiencies. ------------------------------------ >
  // ----------------------------------------------------------- >

  double S_tbc[325];
  for (int n = 0; n < 325; n++)
  {
     S_tbc[n] = 1.0;
  }

  S_tbc[0] = 2.4*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 15.4*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.75*cs[14] + 3.6*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.83*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[1] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[11] = 2.0*cs[0] + cs[1] + cs[2] + 6.0*cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 3.5*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.5*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[32] = cs[0] + cs[1] + cs[2] + cs[4] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + cs[13] + 0.75*cs[14] + 1.5*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 1.5*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[33] = cs[3];

  S_tbc[34] = cs[5];

  S_tbc[35] = cs[47];

  S_tbc[36] = cs[48];

  S_tbc[38] = cs[1] + cs[2] + cs[3] + cs[4] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + cs[14] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.63*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[39] = cs[0];

  S_tbc[40] = cs[5];

  S_tbc[41] = cs[15];

  S_tbc[42] = 0.73*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 3.65*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + cs[14] + cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.38*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[49] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[51] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 3.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[53] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[55] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[56] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[58] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[62] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[69] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[70] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[71] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[73] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[75] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[82] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[84] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[94] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[130] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[139] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[146] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[157] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[165] = cs[5];

  S_tbc[166] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[173] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[184] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.625*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[186] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[204] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[211] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[226] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[229] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[236] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[240] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[268] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[288] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[302] = cs[12];

  S_tbc[303] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[311] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[317] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  S_tbc[319] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + 6.0*cs[5] + cs[6] + cs[7] + cs[8] + cs[9] + cs[10] + cs[11] + cs[12] + 2.0*cs[13] + 1.5*cs[14] + 2.0*cs[15] + cs[16] + cs[17] + cs[18] + cs[19] + cs[20] + cs[21] + cs[22] + cs[23] + cs[24] + cs[25] + 3.0*cs[26] + cs[27] + cs[28] + cs[29] + cs[30] + cs[31] + cs[32] + cs[33] + cs[34] + cs[35] + cs[36] + cs[37] + cs[38] + cs[39] + cs[40] + cs[41] + cs[42] + cs[43] + cs[44] + cs[45] + cs[46] + cs[47] + 0.7*cs[48] + cs[49] + cs[50] + cs[51] + cs[52];

  // ----------------------------------------------------------- >
  // Gibbs energy. --------------------------------------------- >
  // ----------------------------------------------------------- >

  int m;
  double hi,scs;
  double gbs[53];

  for (int n=0; n<=52; n++)
  {
    m = ( T <= th.NASA7(n,0) ) ? 8 : 1;

    hi     = th.NASA7(n,m+0)                  +
             th.NASA7(n,m+1)*    T      / 2.0 +
             th.NASA7(n,m+2)*pow(T,2.0) / 3.0 +
             th.NASA7(n,m+3)*pow(T,3.0) / 4.0 +
             th.NASA7(n,m+4)*pow(T,4.0) / 5.0 +
             th.NASA7(n,m+5)/    T            ;
    scs    = th.NASA7(n,m+0)*log(T)           +
             th.NASA7(n,m+1)*    T            +
             th.NASA7(n,m+2)*pow(T,2.0) / 2.0 +
             th.NASA7(n,m+3)*pow(T,3.0) / 3.0 +
             th.NASA7(n,m+4)*pow(T,4.0) / 4.0 +
             th.NASA7(n,m+6)                  ;

    gbs[n] = hi-scs                         ;
  }

  // ----------------------------------------------------------- >
  // Rate Constants. ------------------------------------------- >
  // FallOff Modifications. ------------------------------------ >
  // Forward, backward, net rates of progress. ----------------- >
  // ----------------------------------------------------------- >

  double k_f, dG, K_c; 

  double Fcent;
  double pmod;
  double Pr,k0;
  double A,f1,F_pdr;
  double C,N;

  double q_f, q_b;
  double q[325];

  // Reaction #0
  k_f = exp(log(120000000000.00002)-1.0*logT);
   dG =  -2.0*gbs[2] + gbs[3];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #0
  q_f =   S_tbc[0] * k_f * pow(cs[2],2.0);
  q_b = - S_tbc[0] * k_f/K_c * cs[3];
  q[0] =   q_f + q_b;

  // Reaction #1
  k_f = exp(log(500000000000.0001)-1.0*logT);
   dG =  - gbs[1] - gbs[2] + gbs[4];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #1
  q_f =   S_tbc[1] * k_f * cs[1] * cs[2];
  q_b = - S_tbc[1] * k_f/K_c * cs[4];
  q[1] =   q_f + q_b;

  // Reaction #2
  k_f = exp(log(38.7)+2.7*logT-(3150.1542797022735/T));
   dG =  - gbs[0] + gbs[1] - gbs[2] + gbs[4];
  K_c = exp(-dG);
  q_f =   S_tbc[2] * k_f * cs[0] * cs[2];
  q_b = - S_tbc[2] * k_f/K_c * cs[1] * cs[4];
  q[2] =   q_f + q_b;

  // Reaction #3
  k_f = 20000000000.000004;
   dG =  - gbs[2] + gbs[3] + gbs[4] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[3] * k_f * cs[2] * cs[6];
  q_b = - S_tbc[3] * k_f/K_c * cs[3] * cs[4];
  q[3] =   q_f + q_b;

  // Reaction #4
  k_f = exp(log(9630.0)+2*logT-(2012.8781339950629/T));
   dG =  - gbs[2] + gbs[4] + gbs[6] - gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[4] * k_f * cs[2] * cs[7];
  q_b = - S_tbc[4] * k_f/K_c * cs[4] * cs[6];
  q[4] =   q_f + q_b;

  // Reaction #5
  k_f = 57000000000.00001;
   dG =   gbs[1] - gbs[2] - gbs[9] + gbs[14];
  K_c = exp(-dG);
  q_f =   S_tbc[5] * k_f * cs[2] * cs[9];
  q_b = - S_tbc[5] * k_f/K_c * cs[1] * cs[14];
  q[5] =   q_f + q_b;

  // Reaction #6
  k_f = 80000000000.00002;
   dG =   gbs[1] - gbs[2] - gbs[10] + gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[6] * k_f * cs[2] * cs[10];
  q_b = - S_tbc[6] * k_f/K_c * cs[1] * cs[16];
  q[6] =   q_f + q_b;

  // Reaction #7
  k_f = 15000000000.000002;
   dG =   gbs[0] - gbs[2] - gbs[11] + gbs[14];
  K_c = exp(-dG);
  q_f =   S_tbc[7] * k_f * cs[2] * cs[11];
  q_b = - S_tbc[7] * k_f/K_c * cs[0] * cs[14];
  q[7] =   q_f + q_b;

  // Reaction #8
  k_f = 15000000000.000002;
   dG =   gbs[1] - gbs[2] - gbs[11] + gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[8] * k_f * cs[2] * cs[11];
  q_b = - S_tbc[8] * k_f/K_c * cs[1] * cs[16];
  q[8] =   q_f + q_b;

  // Reaction #9
  k_f = 50600000000.00001;
   dG =   gbs[1] - gbs[2] - gbs[12] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[9] * k_f * cs[2] * cs[12];
  q_b = - S_tbc[9] * k_f/K_c * cs[1] * cs[17];
  q[9] =   q_f + q_b;

  // Reaction #10
  k_f = exp(log(1020000.0000000001)+1.5*logT-(4327.687988089386/T));
   dG =  - gbs[2] + gbs[4] + gbs[12] - gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[10] * k_f * cs[2] * cs[13];
  q_b = - S_tbc[10] * k_f/K_c * cs[4] * cs[12];
  q[10] =   q_f + q_b;

  // Reaction #11
  k_f = exp(log(18000000.000000004)-(1200.1785873945562/T));
   dG =  - gbs[2] - gbs[14] + gbs[15];
  K_c = exp(-dG)/prefRuT;
  //  Lindeman Reaction #11
  Fcent = 1.0;
  k0 = exp(log(602000000.0000001)-(1509.6586004962971/T));
  Pr = S_tbc[11]*k0/k_f;
  pmod = Pr/(1.0 + Pr);
  k_f = k_f*pmod;
  q_f =   k_f * cs[2] * cs[14];
  q_b = - k_f/K_c * cs[15];
  q[11] =   q_f + q_b;

  // Reaction #12
  k_f = 30000000000.000004;
   dG =  - gbs[2] + gbs[4] + gbs[14] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[12] * k_f * cs[2] * cs[16];
  q_b = - S_tbc[12] * k_f/K_c * cs[4] * cs[14];
  q[12] =   q_f + q_b;

  // Reaction #13
  k_f = 30000000000.000004;
   dG =   gbs[1] - gbs[2] + gbs[15] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[13] * k_f * cs[2] * cs[16];
  q_b = - S_tbc[13] * k_f/K_c * cs[1] * cs[15];
  q[13] =   q_f + q_b;

  // Reaction #14
  k_f = exp(log(39000000000.00001)-(1781.3971485856307/T));
   dG =  - gbs[2] + gbs[4] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[14] * k_f * cs[2] * cs[17];
  q_b = - S_tbc[14] * k_f/K_c * cs[4] * cs[16];
  q[14] =   q_f + q_b;

  // Reaction #15
  k_f = 10000000000.000002;
   dG =  - gbs[2] + gbs[4] + gbs[17] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[15] * k_f * cs[2] * cs[18];
  q_b = - S_tbc[15] * k_f/K_c * cs[4] * cs[17];
  q[15] =   q_f + q_b;

  // Reaction #16
  k_f = 10000000000.000002;
   dG =  - gbs[2] + gbs[4] + gbs[17] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[16] * k_f * cs[2] * cs[19];
  q_b = - S_tbc[16] * k_f/K_c * cs[4] * cs[17];
  q[16] =   q_f + q_b;

  // Reaction #17
  k_f = exp(log(388.00000000000006)+2.5*logT-(1559.9805538461737/T));
   dG =  - gbs[2] + gbs[4] + gbs[18] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[17] * k_f * cs[2] * cs[20];
  q_b = - S_tbc[17] * k_f/K_c * cs[4] * cs[18];
  q[17] =   q_f + q_b;

  // Reaction #18
  k_f = exp(log(130.00000000000003)+2.5*logT-(2516.097667493829/T));
   dG =  - gbs[2] + gbs[4] + gbs[19] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[18] * k_f * cs[2] * cs[20];
  q_b = - S_tbc[18] * k_f/K_c * cs[4] * cs[19];
  q[18] =   q_f + q_b;

  // Reaction #19
  k_f = 50000000000.00001;
   dG =  - gbs[2] + gbs[9] + gbs[14] - gbs[21];
  K_c = exp(-dG);
  q_f =   S_tbc[19] * k_f * cs[2] * cs[21];
  q_b = - S_tbc[19] * k_f/K_c * cs[9] * cs[14];
  q[19] =   q_f + q_b;

  // Reaction #20
  k_f = exp(log(13500.000000000002)+2*logT-(956.117113647655/T));
   dG =   gbs[1] - gbs[2] - gbs[22] + gbs[27];
  K_c = exp(-dG);
  q_f =   S_tbc[20] * k_f * cs[2] * cs[22];
  q_b = - S_tbc[20] * k_f/K_c * cs[1] * cs[27];
  q[20] =   q_f + q_b;

  // Reaction #21
  k_f = exp(log(4.600000000000001e+16)-1.41*logT-(14568.205494789268/T));
   dG =  - gbs[2] + gbs[4] + gbs[21] - gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[21] * k_f * cs[2] * cs[22];
  q_b = - S_tbc[21] * k_f/K_c * cs[4] * cs[21];
  q[21] =   q_f + q_b;

  // Reaction #22
  k_f = exp(log(6940.000000000001)+2*logT-(956.117113647655/T));
   dG =  - gbs[2] + gbs[10] + gbs[14] - gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[22] * k_f * cs[2] * cs[22];
  q_b = - S_tbc[22] * k_f/K_c * cs[10] * cs[14];
  q[22] =   q_f + q_b;

  // Reaction #23
  k_f = 30000000000.000004;
   dG =   gbs[1] - gbs[2] - gbs[23] + gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[23] * k_f * cs[2] * cs[23];
  q_b = - S_tbc[23] * k_f/K_c * cs[1] * cs[28];
  q[23] =   q_f + q_b;

  // Reaction #24
  k_f = exp(log(12500.000000000002)+1.83*logT-(110.70829736972846/T));
   dG =  - gbs[2] + gbs[12] + gbs[16] - gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[24] * k_f * cs[2] * cs[24];
  q_b = - S_tbc[24] * k_f/K_c * cs[12] * cs[16];
  q[24] =   q_f + q_b;

  // Reaction #25
  k_f = 22400000000.000004;
   dG =  - gbs[2] + gbs[12] + gbs[17] - gbs[25];
  K_c = exp(-dG);
  q_f =   S_tbc[25] * k_f * cs[2] * cs[25];
  q_b = - S_tbc[25] * k_f/K_c * cs[12] * cs[17];
  q[25] =   q_f + q_b;

  // Reaction #26
  k_f = exp(log(89800.00000000001)+1.92*logT-(2863.319145607977/T));
   dG =  - gbs[2] + gbs[4] + gbs[25] - gbs[26];
  K_c = exp(-dG);
  q_f =   S_tbc[26] * k_f * cs[2] * cs[26];
  q_b = - S_tbc[26] * k_f/K_c * cs[4] * cs[25];
  q[26] =   q_f + q_b;

  // Reaction #27
  k_f = 100000000000.00002;
   dG =   gbs[1] - gbs[2] +2.0*gbs[14] - gbs[27];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[27] * k_f * cs[2] * cs[27];
  q_b = - S_tbc[27] * k_f/K_c * cs[1] * pow(cs[14],2.0);
  q[27] =   q_f + q_b;

  // Reaction #28
  k_f = exp(log(10000000000.000002)-(4025.7562679901257/T));
   dG =  - gbs[2] + gbs[4] + gbs[27] - gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[28] * k_f * cs[2] * cs[28];
  q_b = - S_tbc[28] * k_f/K_c * cs[4] * cs[27];
  q[28] =   q_f + q_b;

  // Reaction #29
  k_f = exp(log(1750000000.0000002)-(679.3463702233338/T));
   dG =  - gbs[2] + gbs[10] + gbs[15] - gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[29] * k_f * cs[2] * cs[28];
  q_b = - S_tbc[29] * k_f/K_c * cs[10] * cs[15];
  q[29] =   q_f + q_b;

  // Reaction #30
  k_f = exp(log(2500000000.0000005)-(24053.893701241002/T));
   dG =   gbs[2] - gbs[3] - gbs[14] + gbs[15];
  K_c = exp(-dG);
  q_f =   S_tbc[30] * k_f * cs[3] * cs[14];
  q_b = - S_tbc[30] * k_f/K_c * cs[2] * cs[15];
  q[30] =   q_f + q_b;

  // Reaction #31
  k_f = exp(log(100000000000.00002)-(20128.78133995063/T));
   dG =  - gbs[3] + gbs[6] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[31] * k_f * cs[3] * cs[17];
  q_b = - S_tbc[31] * k_f/K_c * cs[6] * cs[16];
  q[31] =   q_f + q_b;

  // Reaction #32
  k_f = exp(log(2800000000000.0005)-0.86*logT);
   dG =  - gbs[1] - gbs[3] + gbs[6];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #32
  q_f =   S_tbc[32] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[32] * k_f/K_c * cs[6];
  q[32] =   q_f + q_b;

  // Reaction #33
  k_f = exp(log(20800000000000.004)-1.24*logT);
   dG =  - gbs[1] - gbs[3] + gbs[6];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #33
  q_f =   S_tbc[33] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[33] * k_f/K_c * cs[6];
  q[33] =   q_f + q_b;

  // Reaction #34
  k_f = exp(log(11260000000000.002)-0.76*logT);
   dG =  - gbs[1] - gbs[3] + gbs[6];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #34
  q_f =   S_tbc[34] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[34] * k_f/K_c * cs[6];
  q[34] =   q_f + q_b;

  // Reaction #35
  k_f = exp(log(26000000000000.004)-1.24*logT);
   dG =  - gbs[1] - gbs[3] + gbs[6];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #35
  q_f =   S_tbc[35] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[35] * k_f/K_c * cs[6];
  q[35] =   q_f + q_b;

  // Reaction #36
  k_f = exp(log(700000000000.0001)-0.8*logT);
   dG =  - gbs[1] - gbs[3] + gbs[6];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #36
  q_f =   S_tbc[36] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[36] * k_f/K_c * cs[6];
  q[36] =   q_f + q_b;

  // Reaction #37
  k_f = exp(log(26500000000000.004)-0.6707*logT-(8575.364070352467/T));
   dG =  - gbs[1] + gbs[2] - gbs[3] + gbs[4];
  K_c = exp(-dG);
  q_f =   S_tbc[37] * k_f * cs[1] * cs[3];
  q_b = - S_tbc[37] * k_f/K_c * cs[2] * cs[4];
  q[37] =   q_f + q_b;

  // Reaction #38
  k_f = exp(log(1000000000000.0002)-1.0*logT);
   dG =   gbs[0] -2.0*gbs[1];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #38
  q_f =   S_tbc[38] * k_f * pow(cs[1],2.0);
  q_b = - S_tbc[38] * k_f/K_c * cs[0];
  q[38] =   q_f + q_b;

  // Reaction #39
  k_f = exp(log(90000000000.00002)-0.6*logT);
   dG =   gbs[0] -2.0*gbs[1];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #39
  q_f =   S_tbc[39] * k_f * pow(cs[1],2.0);
  q_b = - S_tbc[39] * k_f/K_c * cs[0];
  q[39] =   q_f + q_b;

  // Reaction #40
  k_f = exp(log(60000000000000.01)-1.25*logT);
   dG =   gbs[0] -2.0*gbs[1];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #40
  q_f =   S_tbc[40] * k_f * pow(cs[1],2.0);
  q_b = - S_tbc[40] * k_f/K_c * cs[0];
  q[40] =   q_f + q_b;

  // Reaction #41
  k_f = exp(log(550000000000000.1)-2.0*logT);
   dG =   gbs[0] -2.0*gbs[1];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #41
  q_f =   S_tbc[41] * k_f * pow(cs[1],2.0);
  q_b = - S_tbc[41] * k_f/K_c * cs[0];
  q[41] =   q_f + q_b;

  // Reaction #42
  k_f = exp(log(2.2000000000000004e+16)-2.0*logT);
   dG =  - gbs[1] - gbs[4] + gbs[5];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #42
  q_f =   S_tbc[42] * k_f * cs[1] * cs[4];
  q_b = - S_tbc[42] * k_f/K_c * cs[5];
  q[42] =   q_f + q_b;

  // Reaction #43
  k_f = exp(log(3970000000.0000005)-(337.66030697767184/T));
   dG =  - gbs[1] + gbs[2] + gbs[5] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[43] * k_f * cs[1] * cs[6];
  q_b = - S_tbc[43] * k_f/K_c * cs[2] * cs[5];
  q[43] =   q_f + q_b;

  // Reaction #44
  k_f = exp(log(44800000000.00001)-(537.4384617766818/T));
   dG =   gbs[0] - gbs[1] + gbs[3] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[44] * k_f * cs[1] * cs[6];
  q_b = - S_tbc[44] * k_f/K_c * cs[0] * cs[3];
  q[44] =   q_f + q_b;

  // Reaction #45
  k_f = exp(log(84000000000.00002)-(319.54440377171625/T));
   dG =  - gbs[1] +2.0*gbs[4] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[45] * k_f * cs[1] * cs[6];
  q_b = - S_tbc[45] * k_f/K_c * pow(cs[4],2.0);
  q[45] =   q_f + q_b;

  // Reaction #46
  k_f = exp(log(12100.000000000002)+2*logT-(2616.741574193582/T));
   dG =   gbs[0] - gbs[1] + gbs[6] - gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[46] * k_f * cs[1] * cs[7];
  q_b = - S_tbc[46] * k_f/K_c * cs[0] * cs[6];
  q[46] =   q_f + q_b;

  // Reaction #47
  k_f = exp(log(10000000000.000002)-(1811.5903205955567/T));
   dG =  - gbs[1] + gbs[4] + gbs[5] - gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[47] * k_f * cs[1] * cs[7];
  q_b = - S_tbc[47] * k_f/K_c * cs[4] * cs[5];
  q[47] =   q_f + q_b;

  // Reaction #48
  k_f = 165000000000.00003;
   dG =   gbs[0] - gbs[1] + gbs[8] - gbs[9];
  K_c = exp(-dG);
  q_f =   S_tbc[48] * k_f * cs[1] * cs[9];
  q_b = - S_tbc[48] * k_f/K_c * cs[0] * cs[8];
  q[48] =   q_f + q_b;

  // Reaction #49
  k_f = 600000000000.0001;
   dG =  - gbs[1] - gbs[10] + gbs[12];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #49
  Fcent = (1.0 - (0.562))*exp(-T/(91.0)) + (0.562) *exp(-T/(5836.0)) + exp(-(8552.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.0400000000000002e+20)-2.76*logT-(805.1512535980252/T));
  Pr = S_tbc[49]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[10];
  q_b = - k_f/K_c * cs[12];
  q[49] =   q_f + q_b;

  // Reaction #50
  k_f = 30000000000.000004;
   dG =   gbs[0] - gbs[1] + gbs[9] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[50] * k_f * cs[1] * cs[11];
  q_b = - S_tbc[50] * k_f/K_c * cs[0] * cs[9];
  q[50] =   q_f + q_b;

  // Reaction #51
  k_f = exp(log(13900000000000.002)-0.534*logT-(269.72566995533845/T));
   dG =  - gbs[1] - gbs[12] + gbs[13];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #51
  Fcent = (1.0 - (0.783))*exp(-T/(74.0)) + (0.783) *exp(-T/(2941.0)) + exp(-(6964.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.6200000000000006e+27)-4.76*logT-(1227.8556617369884/T));
  Pr = S_tbc[51]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[12];
  q_b = - k_f/K_c * cs[13];
  q[51] =   q_f + q_b;

  // Reaction #52
  k_f = exp(log(660000.0000000001)+1.62*logT-(5454.899743126621/T));
   dG =   gbs[0] - gbs[1] + gbs[12] - gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[52] * k_f * cs[1] * cs[13];
  q_b = - S_tbc[52] * k_f/K_c * cs[0] * cs[12];
  q[52] =   q_f + q_b;

  // Reaction #53
  k_f = exp(log(1090000000.0000002)+0.48*logT-(-130.8370787096791/T));
   dG =  - gbs[1] - gbs[16] + gbs[17];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #53
  Fcent = (1.0 - (0.7824))*exp(-T/(271.0)) + (0.7824) *exp(-T/(2755.0)) + exp(-(6570.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.4700000000000005e+18)-2.57*logT-(213.86830173697544/T));
  Pr = S_tbc[53]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[16];
  q_b = - k_f/K_c * cs[17];
  q[53] =   q_f + q_b;

  // Reaction #54
  k_f = 73400000000.00002;
   dG =   gbs[0] - gbs[1] + gbs[14] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[54] * k_f * cs[1] * cs[16];
  q_b = - S_tbc[54] * k_f/K_c * cs[0] * cs[14];
  q[54] =   q_f + q_b;

  // Reaction #55
  k_f = exp(log(540000000.0000001)+0.454*logT-(1811.5903205955567/T));
   dG =  - gbs[1] - gbs[17] + gbs[18];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #55
  Fcent = (1.0 - (0.7187))*exp(-T/(103.00000000000001)) + (0.7187) *exp(-T/(1291.0)) + exp(-(4160.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.2700000000000002e+26)-4.82*logT-(3286.0235537469403/T));
  Pr = S_tbc[55]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[17];
  q_b = - k_f/K_c * cs[18];
  q[55] =   q_f + q_b;

  // Reaction #56
  k_f = exp(log(540000000.0000001)+0.454*logT-(1308.370787096791/T));
   dG =  - gbs[1] - gbs[17] + gbs[19];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #56
  Fcent = (1.0 - (0.758))*exp(-T/(94.0)) + (0.758) *exp(-T/(1555.0)) + exp(-(4200.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.2000000000000006e+24)-4.8*logT-(2797.9006062531375/T));
  Pr = S_tbc[56]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[17];
  q_b = - k_f/K_c * cs[19];
  q[56] =   q_f + q_b;

  // Reaction #57
  k_f = exp(log(57400.000000000015)+1.9*logT-(1379.8279608536157/T));
   dG =   gbs[0] - gbs[1] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[57] * k_f * cs[1] * cs[17];
  q_b = - S_tbc[57] * k_f/K_c * cs[0] * cs[16];
  q[57] =   q_f + q_b;

  // Reaction #58
  k_f = exp(log(1055000000.0000002)+0.5*logT-(43.27687988089385/T));
   dG =  - gbs[1] - gbs[18] + gbs[20];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #58
  Fcent = (1.0 - (0.6))*exp(-T/(100.0)) + (0.6) *exp(-T/(90000.0)) + exp(-(10000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(4.360000000000001e+25)-4.65*logT-(2556.35523017373/T));
  Pr = S_tbc[58]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[18];
  q_b = - k_f/K_c * cs[20];
  q[58] =   q_f + q_b;

  // Reaction #59
  k_f = 20000000000.000004;
   dG =   gbs[0] - gbs[1] + gbs[17] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[59] * k_f * cs[1] * cs[18];
  q_b = - S_tbc[59] * k_f/K_c * cs[0] * cs[17];
  q[59] =   q_f + q_b;

  // Reaction #60
  k_f = exp(log(165000000.00000003)+0.65*logT-(-142.91434751364946/T));
   dG =  - gbs[1] + gbs[4] + gbs[12] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[60] * k_f * cs[1] * cs[18];
  q_b = - S_tbc[60] * k_f/K_c * cs[4] * cs[12];
  q[60] =   q_f + q_b;

  // Reaction #61
  k_f = exp(log(32800000000.000004)-0.09*logT-(306.9639154342471/T));
   dG =  - gbs[1] + gbs[5] + gbs[11] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[61] * k_f * cs[1] * cs[18];
  q_b = - S_tbc[61] * k_f/K_c * cs[5] * cs[11];
  q[61] =   q_f + q_b;

  // Reaction #62
  k_f = exp(log(2430000000.0000005)+0.515*logT-(25.160976674938286/T));
   dG =  - gbs[1] - gbs[19] + gbs[20];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #62
  Fcent = (1.0 - (0.7))*exp(-T/(100.0)) + (0.7) *exp(-T/(90000.0)) + exp(-(10000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(4.660000000000001e+35)-7.44*logT-(7085.331031662621/T));
  Pr = S_tbc[62]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[19];
  q_b = - k_f/K_c * cs[20];
  q[62] =   q_f + q_b;

  // Reaction #63
  k_f = exp(log(41500.00000000001)+1.63*logT-(968.1943824516253/T));
   dG =   gbs[18] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[63] * k_f * cs[1] * cs[19];
  q_b = - S_tbc[63] * k_f/K_c * cs[1] * cs[18];
  q[63] =   q_f + q_b;

  // Reaction #64
  k_f = 20000000000.000004;
   dG =   gbs[0] - gbs[1] + gbs[17] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[64] * k_f * cs[1] * cs[19];
  q_b = - S_tbc[64] * k_f/K_c * cs[0] * cs[17];
  q[64] =   q_f + q_b;

  // Reaction #65
  k_f = exp(log(1500000000.0000002)+0.5*logT-(-55.35414868486423/T));
   dG =  - gbs[1] + gbs[4] + gbs[12] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[65] * k_f * cs[1] * cs[19];
  q_b = - S_tbc[65] * k_f/K_c * cs[4] * cs[12];
  q[65] =   q_f + q_b;

  // Reaction #66
  k_f = exp(log(262000000000.00003)-0.23*logT-(538.4449008436793/T));
   dG =  - gbs[1] + gbs[5] + gbs[11] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[66] * k_f * cs[1] * cs[19];
  q_b = - S_tbc[66] * k_f/K_c * cs[5] * cs[11];
  q[66] =   q_f + q_b;

  // Reaction #67
  k_f = exp(log(17000.000000000004)+2.1*logT-(2450.679128138989/T));
   dG =   gbs[0] - gbs[1] + gbs[18] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[67] * k_f * cs[1] * cs[20];
  q_b = - S_tbc[67] * k_f/K_c * cs[0] * cs[18];
  q[67] =   q_f + q_b;

  // Reaction #68
  k_f = exp(log(4200.000000000001)+2.1*logT-(2450.679128138989/T));
   dG =   gbs[0] - gbs[1] + gbs[19] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[68] * k_f * cs[1] * cs[20];
  q_b = - S_tbc[68] * k_f/K_c * cs[0] * cs[19];
  q[68] =   q_f + q_b;

  // Reaction #69
  k_f = exp(log(100000000000000.02)-1.0*logT);
   dG =  - gbs[1] - gbs[21] + gbs[22];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #69
  Fcent = (1.0 - (0.6464))*exp(-T/(132.0)) + (0.6464) *exp(-T/(1315.0)) + exp(-(5566.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(3.750000000000001e+27)-4.8*logT-(956.117113647655/T));
  Pr = S_tbc[69]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[21];
  q_b = - k_f/K_c * cs[22];
  q[69] =   q_f + q_b;

  // Reaction #70
  k_f = exp(log(5600000000.000001)-(1207.7268803970378/T));
   dG =  - gbs[1] - gbs[22] + gbs[23];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #70
  Fcent = (1.0 - (0.7507))*exp(-T/(98.50000000000001)) + (0.7507) *exp(-T/(1302.0)) + exp(-(4167.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(3.8000000000000006e+34)-7.27*logT-(3633.2450318610886/T));
  Pr = S_tbc[70]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[22];
  q_b = - k_f/K_c * cs[23];
  q[70] =   q_f + q_b;

  // Reaction #71
  k_f = exp(log(6080000000.000001)+0.27*logT-(140.9014693796544/T));
   dG =  - gbs[1] - gbs[23] + gbs[24];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #71
  Fcent = (1.0 - (0.782))*exp(-T/(207.49999999999997)) + (0.782) *exp(-T/(2663.0)) + exp(-(6095.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.4000000000000004e+24)-3.86*logT-(1670.6888512159023/T));
  Pr = S_tbc[71]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[23];
  q_b = - k_f/K_c * cs[24];
  q[71] =   q_f + q_b;

  // Reaction #72
  k_f = 30000000000.000004;
   dG =   gbs[0] - gbs[1] + gbs[22] - gbs[23];
  K_c = exp(-dG);
  q_f =   S_tbc[72] * k_f * cs[1] * cs[23];
  q_b = - S_tbc[72] * k_f/K_c * cs[0] * cs[22];
  q[72] =   q_f + q_b;

  // Reaction #73
  k_f = exp(log(540000000.0000001)+0.454*logT-(915.8595509677536/T));
   dG =  - gbs[1] - gbs[24] + gbs[25];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #73
  Fcent = (1.0 - (0.9753))*exp(-T/(209.99999999999997)) + (0.9753) *exp(-T/(983.9999999999999)) + exp(-(4374.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(6.0000000000000005e+35)-7.62*logT-(3507.440148486397/T));
  Pr = S_tbc[73]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[24];
  q_b = - k_f/K_c * cs[25];
  q[73] =   q_f + q_b;

  // Reaction #74
  k_f = exp(log(1325.0000000000002)+2.53*logT-(6159.407090024893/T));
   dG =   gbs[0] - gbs[1] + gbs[23] - gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[74] * k_f * cs[1] * cs[24];
  q_b = - S_tbc[74] * k_f/K_c * cs[0] * cs[23];
  q[74] =   q_f + q_b;

  // Reaction #75
  k_f = exp(log(521000000000000.06)-0.99*logT-(795.0868629280499/T));
   dG =  - gbs[1] - gbs[25] + gbs[26];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #75
  Fcent = (1.0 - (0.8422))*exp(-T/(125.0)) + (0.8422) *exp(-T/(2219.0)) + exp(-(6882.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.9900000000000005e+35)-7.08*logT-(3364.022581439249/T));
  Pr = S_tbc[75]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[25];
  q_b = - k_f/K_c * cs[26];
  q[75] =   q_f + q_b;

  // Reaction #76
  k_f = 2000000000.0000002;
   dG =   gbs[0] - gbs[1] + gbs[24] - gbs[25];
  K_c = exp(-dG);
  q_f =   S_tbc[76] * k_f * cs[1] * cs[25];
  q_b = - S_tbc[76] * k_f/K_c * cs[0] * cs[24];
  q[76] =   q_f + q_b;

  // Reaction #77
  k_f = exp(log(115000.00000000001)+1.9*logT-(3789.243087245706/T));
   dG =   gbs[0] - gbs[1] + gbs[25] - gbs[26];
  K_c = exp(-dG);
  q_f =   S_tbc[77] * k_f * cs[1] * cs[26];
  q_b = - S_tbc[77] * k_f/K_c * cs[0] * cs[25];
  q[77] =   q_f + q_b;

  // Reaction #78
  k_f = 100000000000.00002;
   dG =  - gbs[1] + gbs[11] + gbs[14] - gbs[27];
  K_c = exp(-dG);
  q_f =   S_tbc[78] * k_f * cs[1] * cs[27];
  q_b = - S_tbc[78] * k_f/K_c * cs[11] * cs[14];
  q[78] =   q_f + q_b;

  // Reaction #79
  k_f = exp(log(50000000000.00001)-(4025.7562679901257/T));
   dG =   gbs[0] - gbs[1] + gbs[27] - gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[79] * k_f * cs[1] * cs[28];
  q_b = - S_tbc[79] * k_f/K_c * cs[0] * cs[27];
  q[79] =   q_f + q_b;

  // Reaction #80
  k_f = exp(log(11300000000.000002)-(1725.036560833769/T));
   dG =  - gbs[1] + gbs[12] + gbs[14] - gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[80] * k_f * cs[1] * cs[28];
  q_b = - S_tbc[80] * k_f/K_c * cs[12] * cs[14];
  q[80] =   q_f + q_b;

  // Reaction #81
  k_f = 10000000000.000002;
   dG =   gbs[28] - gbs[29];
  K_c = exp(-dG);
  q_f =   S_tbc[81] * k_f * cs[1] * cs[29];
  q_b = - S_tbc[81] * k_f/K_c * cs[1] * cs[28];
  q[81] =   q_f + q_b;

  // Reaction #82
  k_f = exp(log(43000.00000000001)+1.5*logT-(40056.27486650175/T));
   dG =  - gbs[0] - gbs[14] + gbs[17];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #82
  Fcent = (1.0 - (0.932))*exp(-T/(197.00000000000003)) + (0.932) *exp(-T/(1540.0)) + exp(-(10300.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(5.07e+21)-3.42*logT-(42446.56765062089/T));
  Pr = S_tbc[82]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[0] * cs[14];
  q_b = - k_f/K_c * cs[17];
  q[82] =   q_f + q_b;

  // Reaction #83
  k_f = exp(log(216000.00000000003)+1.51*logT-(1726.0429999007665/T));
   dG =  - gbs[0] + gbs[1] - gbs[4] + gbs[5];
  K_c = exp(-dG);
  q_f =   S_tbc[83] * k_f * cs[0] * cs[4];
  q_b = - S_tbc[83] * k_f/K_c * cs[1] * cs[5];
  q[83] =   q_f + q_b;

  // Reaction #84
  k_f = exp(log(74000000000.00002)-0.37*logT);
   dG =  -2.0*gbs[4] + gbs[7];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #84
  Fcent = (1.0 - (0.7346))*exp(-T/(94.0)) + (0.7346) *exp(-T/(1756.0)) + exp(-(5182.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2300000000000.0005)-0.9*logT-(-855.4732069479018/T));
  Pr = S_tbc[84]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * pow(cs[4],2.0);
  q_b = - k_f/K_c * cs[7];
  q[84] =   q_f + q_b;

  // Reaction #85
  k_f = exp(log(35.7)+2.4*logT-(-1061.7932156823956/T));
   dG =   gbs[2] -2.0*gbs[4] + gbs[5];
  K_c = exp(-dG);
  q_f =   S_tbc[85] * k_f * pow(cs[4],2.0);
  q_b = - S_tbc[85] * k_f/K_c * cs[2] * cs[5];
  q[85] =   q_f + q_b;

  // Reaction #86
  k_f = exp(log(14500000000.000002)-(-251.60976674938286/T));
   dG =   gbs[3] - gbs[4] + gbs[5] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[86] * k_f * cs[4] * cs[6];
  q_b = - S_tbc[86] * k_f/K_c * cs[3] * cs[5];
  q[86] =   q_f + q_b;

  // Reaction #87
  k_f = exp(log(2000000000.0000002)-(214.87474080397297/T));
   dG =  - gbs[4] + gbs[5] + gbs[6] - gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[87] * k_f * cs[4] * cs[7];
  q_b = - S_tbc[87] * k_f/K_c * cs[5] * cs[6];
  q[87] =   q_f + q_b;

  // Reaction #88
  k_f = exp(log(1700000000000000.2)-(14799.686480198701/T));
   dG =  - gbs[4] + gbs[5] + gbs[6] - gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[88] * k_f * cs[4] * cs[7];
  q_b = - S_tbc[88] * k_f/K_c * cs[5] * cs[6];
  q[88] =   q_f + q_b;

  // Reaction #89
  k_f = 50000000000.00001;
   dG =   gbs[1] - gbs[4] - gbs[8] + gbs[14];
  K_c = exp(-dG);
  q_f =   S_tbc[89] * k_f * cs[4] * cs[8];
  q_b = - S_tbc[89] * k_f/K_c * cs[1] * cs[14];
  q[89] =   q_f + q_b;

  // Reaction #90
  k_f = 30000000000.000004;
   dG =   gbs[1] - gbs[4] - gbs[9] + gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[90] * k_f * cs[4] * cs[9];
  q_b = - S_tbc[90] * k_f/K_c * cs[1] * cs[16];
  q[90] =   q_f + q_b;

  // Reaction #91
  k_f = 20000000000.000004;
   dG =   gbs[1] - gbs[4] - gbs[10] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[91] * k_f * cs[4] * cs[10];
  q_b = - S_tbc[91] * k_f/K_c * cs[1] * cs[17];
  q[91] =   q_f + q_b;

  // Reaction #92
  k_f = exp(log(11300.000000000002)+2*logT-(1509.6586004962971/T));
   dG =  - gbs[4] + gbs[5] + gbs[9] - gbs[10];
  K_c = exp(-dG);
  q_f =   S_tbc[92] * k_f * cs[4] * cs[10];
  q_b = - S_tbc[92] * k_f/K_c * cs[5] * cs[9];
  q[92] =   q_f + q_b;

  // Reaction #93
  k_f = 30000000000.000004;
   dG =   gbs[1] - gbs[4] - gbs[11] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[93] * k_f * cs[4] * cs[11];
  q_b = - S_tbc[93] * k_f/K_c * cs[1] * cs[17];
  q[93] =   q_f + q_b;

  // Reaction #94
  k_f = exp(log(2790000000000000.5)-1.43*logT-(669.2819795533584/T));
   dG =  - gbs[4] - gbs[12] + gbs[20];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #94
  Fcent = (1.0 - (0.412))*exp(-T/(195.0)) + (0.412) *exp(-T/(5900.0)) + exp(-(6394.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(4.000000000000001e+30)-5.92*logT-(1580.1093351861243/T));
  Pr = S_tbc[94]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[4] * cs[12];
  q_b = - k_f/K_c * cs[20];
  q[94] =   q_f + q_b;

  // Reaction #95
  k_f = exp(log(56000.00000000001)+1.6*logT-(2727.4498715633104/T));
   dG =  - gbs[4] + gbs[5] + gbs[10] - gbs[12];
  K_c = exp(-dG);
  q_f =   S_tbc[95] * k_f * cs[4] * cs[12];
  q_b = - S_tbc[95] * k_f/K_c * cs[5] * cs[10];
  q[95] =   q_f + q_b;

  // Reaction #96
  k_f = exp(log(644000000000000.1)-1.34*logT-(713.0620789677511/T));
   dG =  - gbs[4] + gbs[5] + gbs[11] - gbs[12];
  K_c = exp(-dG);
  q_f =   S_tbc[96] * k_f * cs[4] * cs[12];
  q_b = - S_tbc[96] * k_f/K_c * cs[5] * cs[11];
  q[96] =   q_f + q_b;

  // Reaction #97
  k_f = exp(log(100000.00000000001)+1.6*logT-(1570.0449445161491/T));
   dG =  - gbs[4] + gbs[5] + gbs[12] - gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[97] * k_f * cs[4] * cs[13];
  q_b = - S_tbc[97] * k_f/K_c * cs[5] * cs[12];
  q[97] =   q_f + q_b;

  // Reaction #98
  k_f = exp(log(47600.00000000001)+1.228*logT-(35.2253673449136/T));
   dG =   gbs[1] - gbs[4] - gbs[14] + gbs[15];
  K_c = exp(-dG);
  q_f =   S_tbc[98] * k_f * cs[4] * cs[14];
  q_b = - S_tbc[98] * k_f/K_c * cs[1] * cs[15];
  q[98] =   q_f + q_b;

  // Reaction #99
  k_f = 50000000000.00001;
   dG =  - gbs[4] + gbs[5] + gbs[14] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[99] * k_f * cs[4] * cs[16];
  q_b = - S_tbc[99] * k_f/K_c * cs[5] * cs[14];
  q[99] =   q_f + q_b;

  // Reaction #100
  k_f = exp(log(3430000.0000000005)+1.18*logT-(-224.9391314739483/T));
   dG =  - gbs[4] + gbs[5] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[100] * k_f * cs[4] * cs[17];
  q_b = - S_tbc[100] * k_f/K_c * cs[5] * cs[16];
  q[100] =   q_f + q_b;

  // Reaction #101
  k_f = 5000000000.000001;
   dG =  - gbs[4] + gbs[5] + gbs[17] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[101] * k_f * cs[4] * cs[18];
  q_b = - S_tbc[101] * k_f/K_c * cs[5] * cs[17];
  q[101] =   q_f + q_b;

  // Reaction #102
  k_f = 5000000000.000001;
   dG =  - gbs[4] + gbs[5] + gbs[17] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[102] * k_f * cs[4] * cs[19];
  q_b = - S_tbc[102] * k_f/K_c * cs[5] * cs[17];
  q[102] =   q_f + q_b;

  // Reaction #103
  k_f = exp(log(1440.0000000000002)+2*logT-(-422.70440813896323/T));
   dG =  - gbs[4] + gbs[5] + gbs[18] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[103] * k_f * cs[4] * cs[20];
  q_b = - S_tbc[103] * k_f/K_c * cs[5] * cs[18];
  q[103] =   q_f + q_b;

  // Reaction #104
  k_f = exp(log(6300.000000000001)+2*logT-(754.8293002481486/T));
   dG =  - gbs[4] + gbs[5] + gbs[19] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[104] * k_f * cs[4] * cs[20];
  q_b = - S_tbc[104] * k_f/K_c * cs[5] * cs[19];
  q[104] =   q_f + q_b;

  // Reaction #105
  k_f = 20000000000.000004;
   dG =   gbs[1] - gbs[4] - gbs[21] + gbs[27];
  K_c = exp(-dG);
  q_f =   S_tbc[105] * k_f * cs[4] * cs[21];
  q_b = - S_tbc[105] * k_f/K_c * cs[1] * cs[27];
  q[105] =   q_f + q_b;

  // Reaction #106
  k_f = exp(log(2.1800000000000005e-07)+4.5*logT-(-503.2195334987657/T));
   dG =   gbs[1] - gbs[4] - gbs[22] + gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[106] * k_f * cs[4] * cs[22];
  q_b = - S_tbc[106] * k_f/K_c * cs[1] * cs[28];
  q[106] =   q_f + q_b;

  // Reaction #107
  k_f = exp(log(504.0000000000001)+2.3*logT-(6793.463702233337/T));
   dG =   gbs[1] - gbs[4] - gbs[22] + gbs[29];
  K_c = exp(-dG);
  q_f =   S_tbc[107] * k_f * cs[4] * cs[22];
  q_b = - S_tbc[107] * k_f/K_c * cs[1] * cs[29];
  q[107] =   q_f + q_b;

  // Reaction #108
  k_f = exp(log(33700.0)+2*logT-(7045.07346898272/T));
   dG =  - gbs[4] + gbs[5] + gbs[21] - gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[108] * k_f * cs[4] * cs[22];
  q_b = - S_tbc[108] * k_f/K_c * cs[5] * cs[21];
  q[108] =   q_f + q_b;

  // Reaction #109
  k_f = exp(log(4.830000000000001e-07)+4*logT-(-1006.4390669975314/T));
   dG =  - gbs[4] + gbs[12] + gbs[14] - gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[109] * k_f * cs[4] * cs[22];
  q_b = - S_tbc[109] * k_f/K_c * cs[12] * cs[14];
  q[109] =   q_f + q_b;

  // Reaction #110
  k_f = 5000000000.000001;
   dG =  - gbs[4] + gbs[5] + gbs[22] - gbs[23];
  K_c = exp(-dG);
  q_f =   S_tbc[110] * k_f * cs[4] * cs[23];
  q_b = - S_tbc[110] * k_f/K_c * cs[5] * cs[22];
  q[110] =   q_f + q_b;

  // Reaction #111
  k_f = exp(log(3600.0000000000005)+2*logT-(1258.0488337469144/T));
   dG =  - gbs[4] + gbs[5] + gbs[23] - gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[111] * k_f * cs[4] * cs[24];
  q_b = - S_tbc[111] * k_f/K_c * cs[5] * cs[23];
  q[111] =   q_f + q_b;

  // Reaction #112
  k_f = exp(log(3540.0000000000005)+2.12*logT-(437.8009941439262/T));
   dG =  - gbs[4] + gbs[5] + gbs[25] - gbs[26];
  K_c = exp(-dG);
  q_f =   S_tbc[112] * k_f * cs[4] * cs[26];
  q_b = - S_tbc[112] * k_f/K_c * cs[5] * cs[25];
  q[112] =   q_f + q_b;

  // Reaction #113
  k_f = exp(log(7500000000.000001)-(1006.4390669975314/T));
   dG =  - gbs[4] + gbs[5] + gbs[27] - gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[113] * k_f * cs[4] * cs[28];
  q_b = - S_tbc[113] * k_f/K_c * cs[5] * cs[27];
  q[113] =   q_f + q_b;

  // Reaction #114
  k_f = exp(log(130000000.00000001)-(-820.2478396029882/T));
   dG =   gbs[3] -2.0*gbs[6] + gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[114] * k_f * pow(cs[6],2.0);
  q_b = - S_tbc[114] * k_f/K_c * cs[3] * cs[7];
  q[114] =   q_f + q_b;

  // Reaction #115
  k_f = exp(log(420000000000.00006)-(6038.634401985189/T));
   dG =   gbs[3] -2.0*gbs[6] + gbs[7];
  K_c = exp(-dG);
  q_f =   S_tbc[115] * k_f * pow(cs[6],2.0);
  q_b = - S_tbc[115] * k_f/K_c * cs[3] * cs[7];
  q[115] =   q_f + q_b;

  // Reaction #116
  k_f = 20000000000.000004;
   dG =   gbs[4] - gbs[6] - gbs[10] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[116] * k_f * cs[6] * cs[10];
  q_b = - S_tbc[116] * k_f/K_c * cs[4] * cs[17];
  q[116] =   q_f + q_b;

  // Reaction #117
  k_f = 1000000000.0000001;
   dG =   gbs[3] - gbs[6] - gbs[12] + gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[117] * k_f * cs[6] * cs[12];
  q_b = - S_tbc[117] * k_f/K_c * cs[3] * cs[13];
  q[117] =   q_f + q_b;

  // Reaction #118
  k_f = 37800000000.00001;
   dG =   gbs[4] - gbs[6] - gbs[12] + gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[118] * k_f * cs[6] * cs[12];
  q_b = - S_tbc[118] * k_f/K_c * cs[4] * cs[19];
  q[118] =   q_f + q_b;

  // Reaction #119
  k_f = exp(log(150000000000.00003)-(11875.980990570872/T));
   dG =   gbs[4] - gbs[6] - gbs[14] + gbs[15];
  K_c = exp(-dG);
  q_f =   S_tbc[119] * k_f * cs[6] * cs[14];
  q_b = - S_tbc[119] * k_f/K_c * cs[4] * cs[15];
  q[119] =   q_f + q_b;

  // Reaction #120
  k_f = exp(log(5600.000000000001)+2*logT-(6038.634401985189/T));
   dG =  - gbs[6] + gbs[7] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[120] * k_f * cs[6] * cs[17];
  q_b = - S_tbc[120] * k_f/K_c * cs[7] * cs[16];
  q[120] =   q_f + q_b;

  // Reaction #121
  k_f = exp(log(58000000000.00001)-(289.85445129528904/T));
   dG =   gbs[2] - gbs[3] - gbs[8] + gbs[14];
  K_c = exp(-dG);
  q_f =   S_tbc[121] * k_f * cs[3] * cs[8];
  q_b = - S_tbc[121] * k_f/K_c * cs[2] * cs[14];
  q[121] =   q_f + q_b;

  // Reaction #122
  k_f = 50000000000.00001;
   dG =   gbs[1] - gbs[8] - gbs[10] + gbs[21];
  K_c = exp(-dG);
  q_f =   S_tbc[122] * k_f * cs[8] * cs[10];
  q_b = - S_tbc[122] * k_f/K_c * cs[1] * cs[21];
  q[122] =   q_f + q_b;

  // Reaction #123
  k_f = 50000000000.00001;
   dG =   gbs[1] - gbs[8] - gbs[12] + gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[123] * k_f * cs[8] * cs[12];
  q_b = - S_tbc[123] * k_f/K_c * cs[1] * cs[22];
  q[123] =   q_f + q_b;

  // Reaction #124
  k_f = 67100000000.00001;
   dG =   gbs[2] - gbs[3] - gbs[9] + gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[124] * k_f * cs[3] * cs[9];
  q_b = - S_tbc[124] * k_f/K_c * cs[2] * cs[16];
  q[124] =   q_f + q_b;

  // Reaction #125
  k_f = exp(log(108000000000.00002)-(1565.0127491811616/T));
   dG =  - gbs[0] + gbs[1] - gbs[9] + gbs[10];
  K_c = exp(-dG);
  q_f =   S_tbc[125] * k_f * cs[0] * cs[9];
  q_b = - S_tbc[125] * k_f/K_c * cs[1] * cs[10];
  q[125] =   q_f + q_b;

  // Reaction #126
  k_f = exp(log(5710000000.000001)-(-379.93074779156814/T));
   dG =   gbs[1] - gbs[5] - gbs[9] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[126] * k_f * cs[5] * cs[9];
  q_b = - S_tbc[126] * k_f/K_c * cs[1] * cs[17];
  q[126] =   q_f + q_b;

  // Reaction #127
  k_f = 40000000000.00001;
   dG =   gbs[1] - gbs[9] - gbs[10] + gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[127] * k_f * cs[9] * cs[10];
  q_b = - S_tbc[127] * k_f/K_c * cs[1] * cs[22];
  q[127] =   q_f + q_b;

  // Reaction #128
  k_f = 30000000000.000004;
   dG =   gbs[1] - gbs[9] - gbs[12] + gbs[23];
  K_c = exp(-dG);
  q_f =   S_tbc[128] * k_f * cs[9] * cs[12];
  q_b = - S_tbc[128] * k_f/K_c * cs[1] * cs[23];
  q[128] =   q_f + q_b;

  // Reaction #129
  k_f = 60000000000.00001;
   dG =   gbs[1] - gbs[9] - gbs[13] + gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[129] * k_f * cs[9] * cs[13];
  q_b = - S_tbc[129] * k_f/K_c * cs[1] * cs[24];
  q[129] =   q_f + q_b;

  // Reaction #130
  k_f = 50000000000.00001;
   dG =  - gbs[9] - gbs[14] + gbs[27];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #130
  Fcent = (1.0 - (0.5757))*exp(-T/(237.00000000000003)) + (0.5757) *exp(-T/(1652.0)) + exp(-(5069.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.6900000000000003e+22)-3.74*logT-(974.2330168536105/T));
  Pr = S_tbc[130]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[9] * cs[14];
  q_b = - k_f/K_c * cs[27];
  q[130] =   q_f + q_b;

  // Reaction #131
  k_f = exp(log(190000000000.00003)-(7946.842873012509/T));
   dG =  - gbs[9] + gbs[14] - gbs[15] + gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[131] * k_f * cs[9] * cs[15];
  q_b = - S_tbc[131] * k_f/K_c * cs[14] * cs[16];
  q[131] =   q_f + q_b;

  // Reaction #132
  k_f = exp(log(94600000000.00002)-(-259.15805975186436/T));
   dG =   gbs[1] - gbs[9] - gbs[17] + gbs[28];
  K_c = exp(-dG);
  q_f =   S_tbc[132] * k_f * cs[9] * cs[17];
  q_b = - S_tbc[132] * k_f/K_c * cs[1] * cs[28];
  q[132] =   q_f + q_b;

  // Reaction #133
  k_f = 50000000000.00001;
   dG =  - gbs[9] + gbs[14] + gbs[22] - gbs[27];
  K_c = exp(-dG);
  q_f =   S_tbc[133] * k_f * cs[9] * cs[27];
  q_b = - S_tbc[133] * k_f/K_c * cs[14] * cs[22];
  q[133] =   q_f + q_b;

  // Reaction #134
  k_f = exp(log(5000000000.000001)-(754.8293002481486/T));
   dG =   gbs[1] - gbs[3] + gbs[4] - gbs[10] + gbs[14];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[134] * k_f * cs[3] * cs[10];
  q_b = - S_tbc[134] * k_f/K_c * cs[1] * cs[4] * cs[14];
  q[134] =   q_f;

  // Reaction #135
  k_f = exp(log(500.0000000000001)+2*logT-(3638.277227196076/T));
   dG =  - gbs[0] + gbs[1] - gbs[10] + gbs[12];
  K_c = exp(-dG);
  q_f =   S_tbc[135] * k_f * cs[0] * cs[10];
  q_b = - S_tbc[135] * k_f/K_c * cs[1] * cs[12];
  q[135] =   q_f + q_b;

  // Reaction #136
  k_f = exp(log(1600000000000.0002)-(6010.454108109258/T));
   dG =   gbs[0] -2.0*gbs[10] + gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[136] * k_f * pow(cs[10],2.0);
  q_b = - S_tbc[136] * k_f/K_c * cs[0] * cs[22];
  q[136] =   q_f + q_b;

  // Reaction #137
  k_f = 40000000000.00001;
   dG =   gbs[1] - gbs[10] - gbs[12] + gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[137] * k_f * cs[10] * cs[12];
  q_b = - S_tbc[137] * k_f/K_c * cs[1] * cs[24];
  q[137] =   q_f + q_b;

  // Reaction #138
  k_f = exp(log(2460.0000000000005)+2*logT-(4161.6255420347925/T));
   dG =  - gbs[10] +2.0*gbs[12] - gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[138] * k_f * cs[10] * cs[13];
  q_b = - S_tbc[138] * k_f/K_c * pow(cs[12],2.0);
  q[138] =   q_f + q_b;

  // Reaction #139
  k_f = exp(log(810000000.0000001)+0.5*logT-(2269.5200960794336/T));
   dG =  - gbs[10] - gbs[14] + gbs[28];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #139
  Fcent = (1.0 - (0.5907))*exp(-T/(275.0)) + (0.5907) *exp(-T/(1226.0)) + exp(-(5185.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.6900000000000006e+27)-5.11*logT-(3570.342590173743/T));
  Pr = S_tbc[139]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[10] * cs[14];
  q_b = - k_f/K_c * cs[28];
  q[139] =   q_f + q_b;

  // Reaction #140
  k_f = 30000000000.000004;
   dG =  - gbs[10] + gbs[14] + gbs[23] - gbs[27];
  K_c = exp(-dG);
  q_f =   S_tbc[140] * k_f * cs[10] * cs[27];
  q_b = - S_tbc[140] * k_f/K_c * cs[14] * cs[23];
  q[140] =   q_f + q_b;

  // Reaction #141
  k_f = exp(log(15000000000.000002)-(301.93172009925945/T));
   dG =   gbs[10] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[141] * k_f * cs[11] * cs[47];
  q_b = - S_tbc[141] * k_f/K_c * cs[10] * cs[47];
  q[141] =   q_f + q_b;

  // Reaction #142
  k_f = exp(log(9000000000.000002)-(301.93172009925945/T));
   dG =   gbs[10] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[142] * k_f * cs[11] * cs[48];
  q_b = - S_tbc[142] * k_f/K_c * cs[10] * cs[48];
  q[142] =   q_f + q_b;

  // Reaction #143
  k_f = 28000000000.000004;
   dG =   gbs[1] - gbs[3] + gbs[4] - gbs[11] + gbs[14];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[143] * k_f * cs[3] * cs[11];
  q_b = - S_tbc[143] * k_f/K_c * cs[1] * cs[4] * cs[14];
  q[143] =   q_f + q_b;

  // Reaction #144
  k_f = 12000000000.000002;
   dG =  - gbs[3] + gbs[5] - gbs[11] + gbs[14];
  K_c = exp(-dG);
  q_f =   S_tbc[144] * k_f * cs[3] * cs[11];
  q_b = - S_tbc[144] * k_f/K_c * cs[5] * cs[14];
  q[144] =   q_f + q_b;

  // Reaction #145
  k_f = 70000000000.00002;
   dG =  - gbs[0] + gbs[1] - gbs[11] + gbs[12];
  K_c = exp(-dG);
  q_f =   S_tbc[145] * k_f * cs[0] * cs[11];
  q_b = - S_tbc[145] * k_f/K_c * cs[1] * cs[12];
  q[145] =   q_f + q_b;

  // Reaction #146
  k_f = exp(log(482000000000000.06)-1.16*logT-(576.1863658560868/T));
   dG =  - gbs[5] - gbs[11] + gbs[20];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #146
  Fcent = (1.0 - (0.6027))*exp(-T/(208.0)) + (0.6027) *exp(-T/(3921.9999999999995)) + exp(-(10180.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.88e+32)-6.36*logT-(2536.226448833779/T));
  Pr = S_tbc[146]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[5] * cs[11];
  q_b = - k_f/K_c * cs[20];
  q[146] =   q_f + q_b;

  // Reaction #147
  k_f = 30000000000.000004;
   dG =   gbs[10] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[147] * k_f * cs[5] * cs[11];
  q_b = - S_tbc[147] * k_f/K_c * cs[5] * cs[10];
  q[147] =   q_f + q_b;

  // Reaction #148
  k_f = exp(log(12000000000.000002)-(-286.83513409429645/T));
   dG =   gbs[1] - gbs[11] - gbs[12] + gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[148] * k_f * cs[11] * cs[12];
  q_b = - S_tbc[148] * k_f/K_c * cs[1] * cs[24];
  q[148] =   q_f + q_b;

  // Reaction #149
  k_f = exp(log(16000000000.000002)-(-286.83513409429645/T));
   dG =  - gbs[11] +2.0*gbs[12] - gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[149] * k_f * cs[11] * cs[13];
  q_b = - S_tbc[149] * k_f/K_c * pow(cs[12],2.0);
  q[149] =   q_f + q_b;

  // Reaction #150
  k_f = 9000000000.000002;
   dG =   gbs[10] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[150] * k_f * cs[11] * cs[14];
  q_b = - S_tbc[150] * k_f/K_c * cs[10] * cs[14];
  q[150] =   q_f + q_b;

  // Reaction #151
  k_f = 7000000000.000001;
   dG =   gbs[10] - gbs[11];
  K_c = exp(-dG);
  q_f =   S_tbc[151] * k_f * cs[11] * cs[15];
  q_b = - S_tbc[151] * k_f/K_c * cs[10] * cs[15];
  q[151] =   q_f + q_b;

  // Reaction #152
  k_f = 14000000000.000002;
   dG =  - gbs[11] + gbs[14] - gbs[15] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[152] * k_f * cs[11] * cs[15];
  q_b = - S_tbc[152] * k_f/K_c * cs[14] * cs[17];
  q[152] =   q_f + q_b;

  // Reaction #153
  k_f = exp(log(40000000000.00001)-(-276.77074342432115/T));
   dG =  - gbs[11] + gbs[12] + gbs[25] - gbs[26];
  K_c = exp(-dG);
  q_f =   S_tbc[153] * k_f * cs[11] * cs[26];
  q_b = - S_tbc[153] * k_f/K_c * cs[12] * cs[25];
  q[153] =   q_f + q_b;

  // Reaction #154
  k_f = exp(log(35600000000.00001)-(15338.13138104238/T));
   dG =   gbs[2] - gbs[3] - gbs[12] + gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[154] * k_f * cs[3] * cs[12];
  q_b = - S_tbc[154] * k_f/K_c * cs[2] * cs[19];
  q[154] =   q_f + q_b;

  // Reaction #155
  k_f = exp(log(2310000000.0000005)-(10222.904823027426/T));
   dG =  - gbs[3] + gbs[4] - gbs[12] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[155] * k_f * cs[3] * cs[12];
  q_b = - S_tbc[155] * k_f/K_c * cs[4] * cs[17];
  q[155] =   q_f + q_b;

  // Reaction #156
  k_f = exp(log(24.500000000000004)+2.47*logT-(2606.6771835236063/T));
   dG =   gbs[6] - gbs[7] - gbs[12] + gbs[13];
  K_c = exp(-dG);
  q_f =   S_tbc[156] * k_f * cs[7] * cs[12];
  q_b = - S_tbc[156] * k_f/K_c * cs[6] * cs[13];
  q[156] =   q_f + q_b;

  // Reaction #157
  k_f = exp(log(67700000000000.01)-1.18*logT-(329.1055749081928/T));
   dG =  -2.0*gbs[12] + gbs[26];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #157
  Fcent = (1.0 - (0.619))*exp(-T/(73.2)) + (0.619) *exp(-T/(1180.0)) + exp(-(9999.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(3.400000000000001e+35)-7.03*logT-(1389.892351523591/T));
  Pr = S_tbc[157]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * pow(cs[12],2.0);
  q_b = - k_f/K_c * cs[26];
  q[157] =   q_f + q_b;

  // Reaction #158
  k_f = exp(log(6840000000.000001)+0.1*logT-(5334.127055086917/T));
   dG =   gbs[1] -2.0*gbs[12] + gbs[25];
  K_c = exp(-dG);
  q_f =   S_tbc[158] * k_f * pow(cs[12],2.0);
  q_b = - S_tbc[158] * k_f/K_c * cs[1] * cs[25];
  q[158] =   q_f + q_b;

  // Reaction #159
  k_f = 26480000000.000004;
   dG =  - gbs[12] + gbs[13] + gbs[14] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[159] * k_f * cs[12] * cs[16];
  q_b = - S_tbc[159] * k_f/K_c * cs[13] * cs[14];
  q[159] =   q_f + q_b;

  // Reaction #160
  k_f = exp(log(3.3200000000000003)+2.81*logT-(2948.866466302767/T));
   dG =  - gbs[12] + gbs[13] + gbs[16] - gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[160] * k_f * cs[12] * cs[17];
  q_b = - S_tbc[160] * k_f/K_c * cs[13] * cs[16];
  q[160] =   q_f + q_b;

  // Reaction #161
  k_f = exp(log(30000.000000000004)+1.5*logT-(5002.002162977731/T));
   dG =  - gbs[12] + gbs[13] + gbs[18] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[161] * k_f * cs[12] * cs[20];
  q_b = - S_tbc[161] * k_f/K_c * cs[13] * cs[18];
  q[161] =   q_f + q_b;

  // Reaction #162
  k_f = exp(log(10000.000000000002)+1.5*logT-(5002.002162977731/T));
   dG =  - gbs[12] + gbs[13] + gbs[19] - gbs[20];
  K_c = exp(-dG);
  q_f =   S_tbc[162] * k_f * cs[12] * cs[20];
  q_b = - S_tbc[162] * k_f/K_c * cs[13] * cs[19];
  q[162] =   q_f + q_b;

  // Reaction #163
  k_f = exp(log(227.00000000000003)+2*logT-(4629.619708188645/T));
   dG =  - gbs[12] + gbs[13] + gbs[23] - gbs[24];
  K_c = exp(-dG);
  q_f =   S_tbc[163] * k_f * cs[12] * cs[24];
  q_b = - S_tbc[163] * k_f/K_c * cs[13] * cs[23];
  q[163] =   q_f + q_b;

  // Reaction #164
  k_f = exp(log(6140.000000000002)+1.74*logT-(5258.644125062102/T));
   dG =  - gbs[12] + gbs[13] + gbs[25] - gbs[26];
  K_c = exp(-dG);
  q_f =   S_tbc[164] * k_f * cs[12] * cs[26];
  q_b = - S_tbc[164] * k_f/K_c * cs[13] * cs[25];
  q[164] =   q_f + q_b;

  // Reaction #165
  k_f = exp(log(1500000000000000.2)-1.0*logT-(8554.732069479018/T));
   dG =   gbs[1] + gbs[14] - gbs[16];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #165
  q_f =   S_tbc[165] * k_f * cs[16];
  q_b = - S_tbc[165] * k_f/K_c * cs[1] * cs[14];
  q[165] =   q_f + q_b;

  // Reaction #166
  k_f = exp(log(187000000000000.03)-1.0*logT-(8554.732069479018/T));
   dG =   gbs[1] + gbs[14] - gbs[16];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #166
  q_f =   S_tbc[166] * k_f * cs[16];
  q_b = - S_tbc[166] * k_f/K_c * cs[1] * cs[14];
  q[166] =   q_f + q_b;

  // Reaction #167
  k_f = exp(log(13450000000.000002)-(201.2878133995063/T));
   dG =  - gbs[3] + gbs[6] + gbs[14] - gbs[16];
  K_c = exp(-dG);
  q_f =   S_tbc[167] * k_f * cs[3] * cs[16];
  q_b = - S_tbc[167] * k_f/K_c * cs[6] * cs[14];
  q[167] =   q_f + q_b;

  // Reaction #168
  k_f = exp(log(18000000000.000004)-(452.8975801488892/T));
   dG =  - gbs[3] + gbs[6] + gbs[17] - gbs[18];
  K_c = exp(-dG);
  q_f =   S_tbc[168] * k_f * cs[3] * cs[18];
  q_b = - S_tbc[168] * k_f/K_c * cs[6] * cs[17];
  q[168] =   q_f + q_b;

  // Reaction #169
  k_f = exp(log(4.2800000000000005e-16)+7.6*logT-(-1776.364953250643/T));
   dG =  - gbs[3] + gbs[6] + gbs[17] - gbs[19];
  K_c = exp(-dG);
  q_f =   S_tbc[169] * k_f * cs[3] * cs[19];
  q_b = - S_tbc[169] * k_f/K_c * cs[6] * cs[17];
  q[169] =   q_f + q_b;

  // Reaction #170
  k_f = exp(log(10000000000.000002)-(-379.93074779156814/T));
   dG =  - gbs[3] + gbs[14] + gbs[16] - gbs[21];
  K_c = exp(-dG);
  q_f =   S_tbc[170] * k_f * cs[3] * cs[21];
  q_b = - S_tbc[170] * k_f/K_c * cs[14] * cs[16];
  q[170] =   q_f + q_b;

  // Reaction #171
  k_f = exp(log(56800000.00000001)+0.9*logT-(1002.9165302630402/T));
   dG =  - gbs[0] + gbs[1] - gbs[21] + gbs[22];
  K_c = exp(-dG);
  q_f =   S_tbc[171] * k_f * cs[0] * cs[21];
  q_b = - S_tbc[171] * k_f/K_c * cs[1] * cs[22];
  q[171] =   q_f + q_b;

  // Reaction #172
  k_f = exp(log(45800000000000.01)-1.39*logT-(510.7678265012472/T));
   dG =  - gbs[3] + gbs[16] + gbs[17] - gbs[23];
  K_c = exp(-dG);
  q_f =   S_tbc[172] * k_f * cs[3] * cs[23];
  q_b = - S_tbc[172] * k_f/K_c * cs[16] * cs[17];
  q[172] =   q_f + q_b;

  // Reaction #173
  k_f = exp(log(8000000000000.0)+0.44*logT-(43664.358921687905/T));
   dG =   gbs[0] + gbs[22] - gbs[24];
  K_c = prefRuT*exp(-dG);
  //  Troe Reaction #173
  Fcent = (1.0 - (0.7345))*exp(-T/(180.0)) + (0.7345) *exp(-T/(1035.0)) + exp(-(5417.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.5800000000000006e+48)-9.3*logT-(49214.870376179286/T));
  Pr = S_tbc[173]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[24];
  q_b = - k_f/K_c * cs[0] * cs[22];
  q[173] =   q_f + q_b;

  // Reaction #174
  k_f = exp(log(840000000.0000001)-(1949.9756923077173/T));
   dG =  - gbs[3] + gbs[6] + gbs[24] - gbs[25];
  K_c = exp(-dG);
  q_f =   S_tbc[174] * k_f * cs[3] * cs[25];
  q_b = - S_tbc[174] * k_f/K_c * cs[6] * cs[24];
  q[174] =   q_f + q_b;

  // Reaction #175
  k_f = exp(log(3200000000.0000005)-(429.74948160794594/T));
   dG =  - gbs[3] + gbs[4] +2.0*gbs[14] - gbs[27];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[175] * k_f * cs[3] * cs[27];
  q_b = - S_tbc[175] * k_f/K_c * cs[4] * pow(cs[14],2.0);
  q[175] =   q_f + q_b;

  // Reaction #176
  k_f = 10000000000.000002;
   dG =  2.0*gbs[14] + gbs[22] -2.0*gbs[27];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[176] * k_f * pow(cs[27],2.0);
  q_b = - S_tbc[176] * k_f/K_c * pow(cs[14],2.0) * cs[22];
  q[176] =   q_f + q_b;

  // Reaction #177
  k_f = exp(log(27000000000.000004)-(178.64293439206185/T));
   dG =   gbs[2] - gbs[30] - gbs[35] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[177] * k_f * cs[30] * cs[35];
  q_b = - S_tbc[177] * k_f/K_c * cs[2] * cs[47];
  q[177] =   q_f + q_b;

  // Reaction #178
  k_f = exp(log(9000000.000000002)+1*logT-(3270.9269677419775/T));
   dG =   gbs[2] - gbs[3] - gbs[30] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[178] * k_f * cs[3] * cs[30];
  q_b = - S_tbc[178] * k_f/K_c * cs[2] * cs[35];
  q[178] =   q_f + q_b;

  // Reaction #179
  k_f = exp(log(33600000000.000008)-(193.73952039702482/T));
   dG =   gbs[1] - gbs[4] - gbs[30] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[179] * k_f * cs[4] * cs[30];
  q_b = - S_tbc[179] * k_f/K_c * cs[1] * cs[35];
  q[179] =   q_f + q_b;

  // Reaction #180
  k_f = exp(log(1400000000.0000002)-(5439.803157121658/T));
   dG =  - gbs[2] + gbs[3] - gbs[37] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[180] * k_f * cs[2] * cs[37];
  q_b = - S_tbc[180] * k_f/K_c * cs[3] * cs[47];
  q[180] =   q_f + q_b;

  // Reaction #181
  k_f = exp(log(29000000000.000004)-(11649.532200496427/T));
   dG =  - gbs[2] +2.0*gbs[35] - gbs[37];
  K_c = exp(-dG);
  q_f =   S_tbc[181] * k_f * cs[2] * cs[37];
  q_b = - S_tbc[181] * k_f/K_c * pow(cs[35],2.0);
  q[181] =   q_f + q_b;

  // Reaction #182
  k_f = exp(log(387000000000.00006)-(9500.784792456698/T));
   dG =  - gbs[1] + gbs[4] - gbs[37] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[182] * k_f * cs[1] * cs[37];
  q_b = - S_tbc[182] * k_f/K_c * cs[4] * cs[47];
  q[182] =   q_f + q_b;

  // Reaction #183
  k_f = exp(log(2000000000.0000002)-(10597.803375484007/T));
   dG =  - gbs[4] + gbs[6] - gbs[37] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[183] * k_f * cs[4] * cs[37];
  q_b = - S_tbc[183] * k_f/K_c * cs[6] * cs[47];
  q[183] =   q_f + q_b;

  // Reaction #184
  k_f = exp(log(79100000000.0)-(28190.358266600855/T));
   dG =   gbs[2] - gbs[37] + gbs[47];
  K_c = prefRuT*exp(-dG);
  //  Lindeman Reaction #184
  Fcent = 1.0;
  k0 = exp(log(637000000000.0001)-(28502.35437737009/T));
  Pr = S_tbc[184]*k0/k_f;
  pmod = Pr/(1.0 + Pr);
  k_f = k_f*pmod;
  q_f =   k_f * cs[37];
  q_b = - k_f/K_c * cs[2] * cs[47];
  q[184] =   q_f + q_b;

  // Reaction #185
  k_f = exp(log(2110000000.0000005)-(-241.54537607940756/T));
   dG =   gbs[4] - gbs[6] - gbs[35] + gbs[36];
  K_c = exp(-dG);
  q_f =   S_tbc[185] * k_f * cs[6] * cs[35];
  q_b = - S_tbc[185] * k_f/K_c * cs[4] * cs[36];
  q[185] =   q_f + q_b;

  // Reaction #186
  k_f = exp(log(106000000000000.02)-1.41*logT);
   dG =  - gbs[2] - gbs[35] + gbs[36];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #186
  q_f =   S_tbc[186] * k_f * cs[2] * cs[35];
  q_b = - S_tbc[186] * k_f/K_c * cs[36];
  q[186] =   q_f + q_b;

  // Reaction #187
  k_f = exp(log(3900000000.0000005)-(-120.77268803970378/T));
   dG =  - gbs[2] + gbs[3] + gbs[35] - gbs[36];
  K_c = exp(-dG);
  q_f =   S_tbc[187] * k_f * cs[2] * cs[36];
  q_b = - S_tbc[187] * k_f/K_c * cs[3] * cs[35];
  q[187] =   q_f + q_b;

  // Reaction #188
  k_f = exp(log(132000000000.00002)-(181.15903205955567/T));
   dG =  - gbs[1] + gbs[4] + gbs[35] - gbs[36];
  K_c = exp(-dG);
  q_f =   S_tbc[188] * k_f * cs[1] * cs[36];
  q_b = - S_tbc[188] * k_f/K_c * cs[4] * cs[35];
  q[188] =   q_f + q_b;

  // Reaction #189
  k_f = 40000000000.00001;
   dG =   gbs[1] - gbs[2] - gbs[31] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[189] * k_f * cs[2] * cs[31];
  q_b = - S_tbc[189] * k_f/K_c * cs[1] * cs[35];
  q[189] =   q_f + q_b;

  // Reaction #190
  k_f = exp(log(32000000000.000004)-(166.0624460545927/T));
   dG =   gbs[0] - gbs[1] + gbs[30] - gbs[31];
  K_c = exp(-dG);
  q_f =   S_tbc[190] * k_f * cs[1] * cs[31];
  q_b = - S_tbc[190] * k_f/K_c * cs[0] * cs[30];
  q[190] =   q_f + q_b;

  // Reaction #191
  k_f = 20000000000.000004;
   dG =   gbs[1] - gbs[4] - gbs[31] + gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[191] * k_f * cs[4] * cs[31];
  q_b = - S_tbc[191] * k_f/K_c * cs[1] * cs[38];
  q[191] =   q_f + q_b;

  // Reaction #192
  k_f = exp(log(2000000.0000000002)+1.2*logT);
   dG =  - gbs[4] + gbs[5] + gbs[30] - gbs[31];
  K_c = exp(-dG);
  q_f =   S_tbc[192] * k_f * cs[4] * cs[31];
  q_b = - S_tbc[192] * k_f/K_c * cs[5] * cs[30];
  q[192] =   q_f + q_b;

  // Reaction #193
  k_f = exp(log(461.00000000000006)+2*logT-(3270.9269677419775/T));
   dG =   gbs[2] - gbs[3] - gbs[31] + gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[193] * k_f * cs[3] * cs[31];
  q_b = - S_tbc[193] * k_f/K_c * cs[2] * cs[38];
  q[193] =   q_f + q_b;

  // Reaction #194
  k_f = exp(log(1280.0000000000002)+1.5*logT-(50.32195334987657/T));
   dG =  - gbs[3] + gbs[4] - gbs[31] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[194] * k_f * cs[3] * cs[31];
  q_b = - S_tbc[194] * k_f/K_c * cs[4] * cs[35];
  q[194] =   q_f + q_b;

  // Reaction #195
  k_f = 15000000000.000002;
   dG =   gbs[1] - gbs[30] - gbs[31] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[195] * k_f * cs[30] * cs[31];
  q_b = - S_tbc[195] * k_f/K_c * cs[1] * cs[47];
  q[195] =   q_f + q_b;

  // Reaction #196
  k_f = exp(log(20000000000.000004)-(6969.590538957906/T));
   dG =   gbs[0] - gbs[5] - gbs[31] + gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[196] * k_f * cs[5] * cs[31];
  q_b = - S_tbc[196] * k_f/K_c * cs[0] * cs[38];
  q[196] =   q_f + q_b;

  // Reaction #197
  k_f = exp(log(21600000000.000004)-0.23*logT);
   dG =   gbs[4] - gbs[31] - gbs[35] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[197] * k_f * cs[31] * cs[35];
  q_b = - S_tbc[197] * k_f/K_c * cs[4] * cs[47];
  q[197] =   q_f + q_b;

  // Reaction #198
  k_f = exp(log(365000000000.00006)-0.45*logT);
   dG =   gbs[1] - gbs[31] - gbs[35] + gbs[37];
  K_c = exp(-dG);
  q_f =   S_tbc[198] * k_f * cs[31] * cs[35];
  q_b = - S_tbc[198] * k_f/K_c * cs[1] * cs[37];
  q[198] =   q_f + q_b;

  // Reaction #199
  k_f = 3000000000.0000005;
   dG =  - gbs[2] + gbs[4] + gbs[31] - gbs[32];
  K_c = exp(-dG);
  q_f =   S_tbc[199] * k_f * cs[2] * cs[32];
  q_b = - S_tbc[199] * k_f/K_c * cs[4] * cs[31];
  q[199] =   q_f + q_b;

  // Reaction #200
  k_f = 39000000000.00001;
   dG =   gbs[1] - gbs[2] - gbs[32] + gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[200] * k_f * cs[2] * cs[32];
  q_b = - S_tbc[200] * k_f/K_c * cs[1] * cs[38];
  q[200] =   q_f + q_b;

  // Reaction #201
  k_f = exp(log(40000000000.00001)-(1836.751297270495/T));
   dG =   gbs[0] - gbs[1] + gbs[31] - gbs[32];
  K_c = exp(-dG);
  q_f =   S_tbc[201] * k_f * cs[1] * cs[32];
  q_b = - S_tbc[201] * k_f/K_c * cs[0] * cs[31];
  q[201] =   q_f + q_b;

  // Reaction #202
  k_f = exp(log(90000.00000000001)+1.5*logT-(-231.48098540943224/T));
   dG =  - gbs[4] + gbs[5] + gbs[31] - gbs[32];
  K_c = exp(-dG);
  q_f =   S_tbc[202] * k_f * cs[4] * cs[32];
  q_b = - S_tbc[202] * k_f/K_c * cs[5] * cs[31];
  q[202] =   q_f + q_b;

  // Reaction #203
  k_f = 330000000.0;
   dG =   gbs[1] - gbs[34] + gbs[47];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[203] * k_f * cs[34];
  q_b = - S_tbc[203] * k_f/K_c * cs[1] * cs[47];
  q[203] =   q_f + q_b;

  // Reaction #204
  k_f = exp(log(130000000000.00002)-0.11*logT-(2506.033276823853/T));
   dG =   gbs[1] - gbs[34] + gbs[47];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #204
  q_f =   S_tbc[204] * k_f * cs[34];
  q_b = - S_tbc[204] * k_f/K_c * cs[1] * cs[47];
  q[204] =   q_f + q_b;

  // Reaction #205
  k_f = 5000000000.000001;
   dG =  - gbs[3] + gbs[6] - gbs[34] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[205] * k_f * cs[3] * cs[34];
  q_b = - S_tbc[205] * k_f/K_c * cs[6] * cs[47];
  q[205] =   q_f + q_b;

  // Reaction #206
  k_f = 25000000000.000004;
   dG =  - gbs[2] + gbs[4] - gbs[34] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[206] * k_f * cs[2] * cs[34];
  q_b = - S_tbc[206] * k_f/K_c * cs[4] * cs[47];
  q[206] =   q_f + q_b;

  // Reaction #207
  k_f = 70000000000.00002;
   dG =  - gbs[2] + gbs[31] - gbs[34] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[207] * k_f * cs[2] * cs[34];
  q_b = - S_tbc[207] * k_f/K_c * cs[31] * cs[35];
  q[207] =   q_f + q_b;

  // Reaction #208
  k_f = 50000000000.00001;
   dG =   gbs[0] - gbs[1] - gbs[34] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[208] * k_f * cs[1] * cs[34];
  q_b = - S_tbc[208] * k_f/K_c * cs[0] * cs[47];
  q[208] =   q_f + q_b;

  // Reaction #209
  k_f = 20000000000.000004;
   dG =  - gbs[4] + gbs[5] - gbs[34] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[209] * k_f * cs[4] * cs[34];
  q_b = - S_tbc[209] * k_f/K_c * cs[5] * cs[47];
  q[209] =   q_f + q_b;

  // Reaction #210
  k_f = 25000000000.000004;
   dG =  - gbs[12] + gbs[13] - gbs[34] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[210] * k_f * cs[12] * cs[34];
  q_b = - S_tbc[210] * k_f/K_c * cs[13] * cs[47];
  q[210] =   q_f + q_b;

  // Reaction #211
  k_f = exp(log(44800000000000.01)-1.32*logT-(372.38245478908664/T));
   dG =  - gbs[1] - gbs[35] + gbs[38];
  K_c = exp(-dG)/prefRuT;
  //  Three Body Reaction #211
  q_f =   S_tbc[211] * k_f * cs[1] * cs[35];
  q_b = - S_tbc[211] * k_f/K_c * cs[38];
  q[211] =   q_f + q_b;

  // Reaction #212
  k_f = 25000000000.000004;
   dG =  - gbs[2] + gbs[4] + gbs[35] - gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[212] * k_f * cs[2] * cs[38];
  q_b = - S_tbc[212] * k_f/K_c * cs[4] * cs[35];
  q[212] =   q_f + q_b;

  // Reaction #213
  k_f = exp(log(900000000.0000001)+0.72*logT-(332.1248921091854/T));
   dG =   gbs[0] - gbs[1] + gbs[35] - gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[213] * k_f * cs[1] * cs[38];
  q_b = - S_tbc[213] * k_f/K_c * cs[0] * cs[35];
  q[213] =   q_f + q_b;

  // Reaction #214
  k_f = exp(log(13000.000000000002)+1.9*logT-(-478.0585568238275/T));
   dG =  - gbs[4] + gbs[5] + gbs[35] - gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[214] * k_f * cs[4] * cs[38];
  q_b = - S_tbc[214] * k_f/K_c * cs[5] * cs[35];
  q[214] =   q_f + q_b;

  // Reaction #215
  k_f = exp(log(10000000000.000002)-(6541.853935483955/T));
   dG =  - gbs[3] + gbs[6] + gbs[35] - gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[215] * k_f * cs[3] * cs[38];
  q_b = - S_tbc[215] * k_f/K_c * cs[6] * cs[35];
  q[215] =   q_f + q_b;

  // Reaction #216
  k_f = 77000000000.00002;
   dG =  - gbs[2] + gbs[14] + gbs[30] - gbs[39];
  K_c = exp(-dG);
  q_f =   S_tbc[216] * k_f * cs[2] * cs[39];
  q_b = - S_tbc[216] * k_f/K_c * cs[14] * cs[30];
  q[216] =   q_f + q_b;

  // Reaction #217
  k_f = 40000000000.00001;
   dG =   gbs[1] - gbs[4] - gbs[39] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[217] * k_f * cs[4] * cs[39];
  q_b = - S_tbc[217] * k_f/K_c * cs[1] * cs[46];
  q[217] =   q_f + q_b;

  // Reaction #218
  k_f = exp(log(8000000000.000001)-(3754.0177199007926/T));
   dG =   gbs[4] - gbs[5] - gbs[39] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[218] * k_f * cs[5] * cs[39];
  q_b = - S_tbc[218] * k_f/K_c * cs[4] * cs[40];
  q[218] =   q_f + q_b;

  // Reaction #219
  k_f = exp(log(6140000000.000001)-(-221.4165947394569/T));
   dG =   gbs[2] - gbs[3] - gbs[39] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[219] * k_f * cs[3] * cs[39];
  q_b = - S_tbc[219] * k_f/K_c * cs[2] * cs[46];
  q[219] =   q_f + q_b;

  // Reaction #220
  k_f = exp(log(295.00000000000006)+2.45*logT-(1127.2117550372352/T));
   dG =  - gbs[0] + gbs[1] - gbs[39] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[220] * k_f * cs[0] * cs[39];
  q_b = - S_tbc[220] * k_f/K_c * cs[1] * cs[40];
  q[220] =   q_f + q_b;

  // Reaction #221
  k_f = 23500000000.000004;
   dG =  - gbs[2] + gbs[14] + gbs[35] - gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[221] * k_f * cs[2] * cs[46];
  q_b = - S_tbc[221] * k_f/K_c * cs[14] * cs[35];
  q[221] =   q_f + q_b;

  // Reaction #222
  k_f = 54000000000.00001;
   dG =  - gbs[1] + gbs[14] + gbs[31] - gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[222] * k_f * cs[1] * cs[46];
  q_b = - S_tbc[222] * k_f/K_c * cs[14] * cs[31];
  q[222] =   q_f + q_b;

  // Reaction #223
  k_f = 2500000000.0000005;
   dG =   gbs[1] - gbs[4] + gbs[14] + gbs[35] - gbs[46];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[223] * k_f * cs[4] * cs[46];
  q_b = - S_tbc[223] * k_f/K_c * cs[1] * cs[14] * cs[35];
  q[223] =   q_f + q_b;

  // Reaction #224
  k_f = 20000000000.000004;
   dG =   gbs[14] - gbs[30] - gbs[46] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[224] * k_f * cs[30] * cs[46];
  q_b = - S_tbc[224] * k_f/K_c * cs[14] * cs[47];
  q[224] =   q_f + q_b;

  // Reaction #225
  k_f = exp(log(2000000000.0000002)-(10064.390669975315/T));
   dG =  - gbs[3] + gbs[15] + gbs[35] - gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[225] * k_f * cs[3] * cs[46];
  q_b = - S_tbc[225] * k_f/K_c * cs[15] * cs[35];
  q[225] =   q_f + q_b;

  // Reaction #226
  k_f = exp(log(310000000000.00006)-(27199.015785608288/T));
   dG =   gbs[14] + gbs[30] - gbs[46];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #226
  q_f =   S_tbc[226] * k_f * cs[46];
  q_b = - S_tbc[226] * k_f/K_c * cs[14] * cs[30];
  q[226] =   q_f + q_b;

  // Reaction #227
  k_f = exp(log(190000000000000.03)-1.52*logT-(372.38245478908664/T));
   dG =   gbs[14] - gbs[35] + gbs[37] - gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[227] * k_f * cs[35] * cs[46];
  q_b = - S_tbc[227] * k_f/K_c * cs[14] * cs[37];
  q[227] =   q_f + q_b;

  // Reaction #228
  k_f = exp(log(3800000000000000.5)-2*logT-(402.5756267990126/T));
   dG =   gbs[15] - gbs[35] - gbs[46] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[228] * k_f * cs[35] * cs[46];
  q_b = - S_tbc[228] * k_f/K_c * cs[15] * cs[47];
  q[228] =   q_f + q_b;

  // Reaction #229
  k_f = exp(log(1.0400000000000003e+26)-3.3*logT-(63707.592940943745/T));
   dG =   gbs[1] + gbs[39] - gbs[40];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #229
  q_f =   S_tbc[229] * k_f * cs[40];
  q_b = - S_tbc[229] * k_f/K_c * cs[1] * cs[39];
  q[229] =   q_f + q_b;

  // Reaction #230
  k_f = exp(log(20.3)+2.64*logT-(2506.033276823853/T));
   dG =   gbs[1] - gbs[2] - gbs[40] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[230] * k_f * cs[2] * cs[40];
  q_b = - S_tbc[230] * k_f/K_c * cs[1] * cs[46];
  q[230] =   q_f + q_b;

  // Reaction #231
  k_f = exp(log(5.07)+2.64*logT-(2506.033276823853/T));
   dG =  - gbs[2] + gbs[14] + gbs[31] - gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[231] * k_f * cs[2] * cs[40];
  q_b = - S_tbc[231] * k_f/K_c * cs[14] * cs[31];
  q[231] =   q_f + q_b;

  // Reaction #232
  k_f = exp(log(3910000.0000000005)+1.58*logT-(13385.639591067169/T));
   dG =  - gbs[2] + gbs[4] + gbs[39] - gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[232] * k_f * cs[2] * cs[40];
  q_b = - S_tbc[232] * k_f/K_c * cs[4] * cs[39];
  q[232] =   q_f + q_b;

  // Reaction #233
  k_f = exp(log(1100.0)+2.03*logT-(6728.045162878498/T));
   dG =   gbs[1] - gbs[4] - gbs[40] + gbs[44];
  K_c = exp(-dG);
  q_f =   S_tbc[233] * k_f * cs[4] * cs[40];
  q_b = - S_tbc[233] * k_f/K_c * cs[1] * cs[44];
  q[233] =   q_f + q_b;

  // Reaction #234
  k_f = exp(log(4.400000000000001)+2.26*logT-(3220.6050143921007/T));
   dG =   gbs[1] - gbs[4] - gbs[40] + gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[234] * k_f * cs[4] * cs[40];
  q_b = - S_tbc[234] * k_f/K_c * cs[1] * cs[45];
  q[234] =   q_f + q_b;

  // Reaction #235
  k_f = exp(log(0.16000000000000003)+2.56*logT-(4528.975801488891/T));
   dG =  - gbs[4] + gbs[14] + gbs[32] - gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[235] * k_f * cs[4] * cs[40];
  q_b = - S_tbc[235] * k_f/K_c * cs[14] * cs[32];
  q[235] =   q_f + q_b;

  // Reaction #236
  k_f = 33000000000.000004;
   dG =  - gbs[1] - gbs[40] + gbs[41];
  K_c = exp(-dG)/prefRuT;
  //  Lindeman Reaction #236
  Fcent = 1.0;
  k0 = exp(log(1.4000000000000003e+20)-3.4*logT-(956.117113647655/T));
  Pr = S_tbc[236]*k0/k_f;
  pmod = Pr/(1.0 + Pr);
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[40];
  q_b = - k_f/K_c * cs[41];
  q[236] =   q_f + q_b;

  // Reaction #237
  k_f = exp(log(60000000000.00001)-(201.2878133995063/T));
   dG =   gbs[10] - gbs[30] - gbs[41] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[237] * k_f * cs[30] * cs[41];
  q_b = - S_tbc[237] * k_f/K_c * cs[10] * cs[47];
  q[237] =   q_f + q_b;

  // Reaction #238
  k_f = exp(log(63000000000.00001)-(23158.1629316132/T));
   dG =  - gbs[8] + gbs[30] + gbs[39] - gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[238] * k_f * cs[8] * cs[47];
  q_b = - S_tbc[238] * k_f/K_c * cs[30] * cs[39];
  q[238] =   q_f + q_b;

  // Reaction #239
  k_f = exp(log(3120000.0000000005)+0.88*logT-(10129.809209330155/T));
   dG =  - gbs[9] + gbs[30] + gbs[40] - gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[239] * k_f * cs[9] * cs[47];
  q_b = - S_tbc[239] * k_f/K_c * cs[30] * cs[40];
  q[239] =   q_f + q_b;

  // Reaction #240
  k_f = exp(log(3100000000.0000005)+0.15*logT);
   dG =  - gbs[9] + gbs[42] - gbs[47];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #240
  Fcent = (1.0 - (0.667))*exp(-T/(235.0)) + (0.667) *exp(-T/(2117.0)) + exp(-(4536.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.3000000000000002e+19)-3.16*logT-(372.38245478908664/T));
  Pr = S_tbc[240]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[9] * cs[47];
  q_b = - k_f/K_c * cs[42];
  q[240] =   q_f + q_b;

  // Reaction #241
  k_f = exp(log(10000000000.000002)-(37238.24547890866/T));
   dG =  - gbs[10] + gbs[31] + gbs[40] - gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[241] * k_f * cs[10] * cs[47];
  q_b = - S_tbc[241] * k_f/K_c * cs[31] * cs[40];
  q[241] =   q_f + q_b;

  // Reaction #242
  k_f = exp(log(100000000.00000001)-(32709.269677419772/T));
   dG =  - gbs[11] + gbs[31] + gbs[40] - gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[242] * k_f * cs[11] * cs[47];
  q_b = - S_tbc[242] * k_f/K_c * cs[31] * cs[40];
  q[242] =   q_f + q_b;

  // Reaction #243
  k_f = 19000000000.000004;
   dG =   gbs[2] - gbs[8] - gbs[35] + gbs[39];
  K_c = exp(-dG);
  q_f =   S_tbc[243] * k_f * cs[8] * cs[35];
  q_b = - S_tbc[243] * k_f/K_c * cs[2] * cs[39];
  q[243] =   q_f + q_b;

  // Reaction #244
  k_f = 29000000000.000004;
   dG =  - gbs[8] + gbs[14] + gbs[30] - gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[244] * k_f * cs[8] * cs[35];
  q_b = - S_tbc[244] * k_f/K_c * cs[14] * cs[30];
  q[244] =   q_f + q_b;

  // Reaction #245
  k_f = 41000000000.00001;
   dG =   gbs[2] - gbs[9] - gbs[35] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[245] * k_f * cs[9] * cs[35];
  q_b = - S_tbc[245] * k_f/K_c * cs[2] * cs[40];
  q[245] =   q_f + q_b;

  // Reaction #246
  k_f = 16200000000.000002;
   dG =   gbs[1] - gbs[9] - gbs[35] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[246] * k_f * cs[9] * cs[35];
  q_b = - S_tbc[246] * k_f/K_c * cs[1] * cs[46];
  q[246] =   q_f + q_b;

  // Reaction #247
  k_f = 24600000000.000004;
   dG =  - gbs[9] + gbs[16] + gbs[30] - gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[247] * k_f * cs[9] * cs[35];
  q_b = - S_tbc[247] * k_f/K_c * cs[16] * cs[30];
  q[247] =   q_f + q_b;

  // Reaction #248
  k_f = exp(log(310000000000000.06)-1.38*logT-(639.0888075434325/T));
   dG =   gbs[1] - gbs[10] - gbs[35] + gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[248] * k_f * cs[10] * cs[35];
  q_b = - S_tbc[248] * k_f/K_c * cs[1] * cs[45];
  q[248] =   q_f + q_b;

  // Reaction #249
  k_f = exp(log(290000000000.00006)-0.69*logT-(382.44684545906193/T));
   dG =   gbs[4] - gbs[10] - gbs[35] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[249] * k_f * cs[10] * cs[35];
  q_b = - S_tbc[249] * k_f/K_c * cs[4] * cs[40];
  q[249] =   q_f + q_b;

  // Reaction #250
  k_f = exp(log(38000000000.00001)-0.36*logT-(291.86732942928415/T));
   dG =   gbs[1] - gbs[10] - gbs[35] + gbs[43];
  K_c = exp(-dG);
  q_f =   S_tbc[250] * k_f * cs[10] * cs[35];
  q_b = - S_tbc[250] * k_f/K_c * cs[1] * cs[43];
  q[250] =   q_f + q_b;

  // Reaction #251
  k_f = exp(log(310000000000000.06)-1.38*logT-(639.0888075434325/T));
   dG =   gbs[1] - gbs[11] - gbs[35] + gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[251] * k_f * cs[11] * cs[35];
  q_b = - S_tbc[251] * k_f/K_c * cs[1] * cs[45];
  q[251] =   q_f + q_b;

  // Reaction #252
  k_f = exp(log(290000000000.00006)-0.69*logT-(382.44684545906193/T));
   dG =   gbs[4] - gbs[11] - gbs[35] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[252] * k_f * cs[11] * cs[35];
  q_b = - S_tbc[252] * k_f/K_c * cs[4] * cs[40];
  q[252] =   q_f + q_b;

  // Reaction #253
  k_f = exp(log(38000000000.00001)-0.36*logT-(291.86732942928415/T));
   dG =   gbs[1] - gbs[11] - gbs[35] + gbs[43];
  K_c = exp(-dG);
  q_f =   S_tbc[253] * k_f * cs[11] * cs[35];
  q_b = - S_tbc[253] * k_f/K_c * cs[1] * cs[43];
  q[253] =   q_f + q_b;

  // Reaction #254
  k_f = exp(log(96000000000.00002)-(14492.722564764454/T));
   dG =   gbs[5] - gbs[12] - gbs[35] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[254] * k_f * cs[12] * cs[35];
  q_b = - S_tbc[254] * k_f/K_c * cs[5] * cs[40];
  q[254] =   q_f + q_b;

  // Reaction #255
  k_f = exp(log(1000000000.0000001)-(10945.024853598155/T));
   dG =   gbs[4] - gbs[12] - gbs[35] + gbs[41];
  K_c = exp(-dG);
  q_f =   S_tbc[255] * k_f * cs[12] * cs[35];
  q_b = - S_tbc[255] * k_f/K_c * cs[4] * cs[41];
  q[255] =   q_f + q_b;

  // Reaction #256
  k_f = 22000000000.000004;
   dG =   gbs[1] - gbs[2] + gbs[14] - gbs[42] + gbs[47];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[256] * k_f * cs[2] * cs[42];
  q_b = - S_tbc[256] * k_f/K_c * cs[1] * cs[14] * cs[47];
  q[256] =   q_f + q_b;

  // Reaction #257
  k_f = 2000000000.0000002;
   dG =  - gbs[2] + gbs[35] + gbs[40] - gbs[42];
  K_c = exp(-dG);
  q_f =   S_tbc[257] * k_f * cs[2] * cs[42];
  q_b = - S_tbc[257] * k_f/K_c * cs[35] * cs[40];
  q[257] =   q_f + q_b;

  // Reaction #258
  k_f = 12000000000.000002;
   dG =   gbs[2] - gbs[3] + gbs[16] - gbs[42] + gbs[47];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[258] * k_f * cs[3] * cs[42];
  q_b = - S_tbc[258] * k_f/K_c * cs[2] * cs[16] * cs[47];
  q[258] =   q_f + q_b;

  // Reaction #259
  k_f = 12000000000.000002;
   dG =   gbs[1] - gbs[4] + gbs[16] - gbs[42] + gbs[47];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[259] * k_f * cs[4] * cs[42];
  q_b = - S_tbc[259] * k_f/K_c * cs[1] * cs[16] * cs[47];
  q[259] =   q_f + q_b;

  // Reaction #260
  k_f = 100000000000.00002;
   dG =  - gbs[1] + gbs[10] - gbs[42] + gbs[47];
  K_c = exp(-dG);
  q_f =   S_tbc[260] * k_f * cs[1] * cs[42];
  q_b = - S_tbc[260] * k_f/K_c * cs[10] * cs[47];
  q[260] =   q_f + q_b;

  // Reaction #261
  k_f = exp(log(98000.00000000001)+1.41*logT-(4277.366034739509/T));
   dG =  - gbs[2] + gbs[15] + gbs[31] - gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[261] * k_f * cs[2] * cs[45];
  q_b = - S_tbc[261] * k_f/K_c * cs[15] * cs[31];
  q[261] =   q_f + q_b;

  // Reaction #262
  k_f = exp(log(150000.00000000003)+1.57*logT-(22141.659473945692/T));
   dG =  - gbs[2] + gbs[14] + gbs[38] - gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[262] * k_f * cs[2] * cs[45];
  q_b = - S_tbc[262] * k_f/K_c * cs[14] * cs[38];
  q[262] =   q_f + q_b;

  // Reaction #263
  k_f = exp(log(2200.0)+2.11*logT-(5736.702681885929/T));
   dG =  - gbs[2] + gbs[4] - gbs[45] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[263] * k_f * cs[2] * cs[45];
  q_b = - S_tbc[263] * k_f/K_c * cs[4] * cs[46];
  q[263] =   q_f + q_b;

  // Reaction #264
  k_f = exp(log(22500.000000000004)+1.7*logT-(1912.23422729531/T));
   dG =  - gbs[1] + gbs[14] + gbs[32] - gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[264] * k_f * cs[1] * cs[45];
  q_b = - S_tbc[264] * k_f/K_c * cs[14] * cs[32];
  q[264] =   q_f + q_b;

  // Reaction #265
  k_f = exp(log(105.00000000000003)+2.5*logT-(6692.8197955335845/T));
   dG =   gbs[0] - gbs[1] - gbs[45] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[265] * k_f * cs[1] * cs[45];
  q_b = - S_tbc[265] * k_f/K_c * cs[0] * cs[46];
  q[265] =   q_f + q_b;

  // Reaction #266
  k_f = exp(log(33000.00000000001)+1.5*logT-(1811.5903205955567/T));
   dG =  - gbs[4] + gbs[5] - gbs[45] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[266] * k_f * cs[4] * cs[45];
  q_b = - S_tbc[266] * k_f/K_c * cs[5] * cs[46];
  q[266] =   q_f + q_b;

  // Reaction #267
  k_f = exp(log(3300.000000000001)+1.5*logT-(1811.5903205955567/T));
   dG =  - gbs[4] + gbs[15] + gbs[32] - gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[267] * k_f * cs[4] * cs[45];
  q_b = - S_tbc[267] * k_f/K_c * cs[15] * cs[32];
  q[267] =   q_f + q_b;

  // Reaction #268
  k_f = exp(log(11800000000000.002)-(42632.75887801543/T));
   dG =   gbs[14] + gbs[31] - gbs[45];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #268
  q_f =   S_tbc[268] * k_f * cs[45];
  q_b = - S_tbc[268] * k_f/K_c * cs[14] * cs[31];
  q[268] =   q_f + q_b;

  // Reaction #269
  k_f = exp(log(2100000000000.0002)-0.69*logT-(1434.1756704714824/T));
   dG =  - gbs[43] + gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[269] * k_f * cs[1] * cs[43];
  q_b = - S_tbc[269] * k_f/K_c * cs[1] * cs[45];
  q[269] =   q_f + q_b;

  // Reaction #270
  k_f = exp(log(270000000.00000006)+0.18*logT-(1066.8254110173834/T));
   dG =  - gbs[1] + gbs[4] + gbs[40] - gbs[43];
  K_c = exp(-dG);
  q_f =   S_tbc[270] * k_f * cs[1] * cs[43];
  q_b = - S_tbc[270] * k_f/K_c * cs[4] * cs[40];
  q[270] =   q_f + q_b;

  // Reaction #271
  k_f = exp(log(170000000000.00003)-0.75*logT-(1454.304451811433/T));
   dG =  - gbs[1] + gbs[14] + gbs[32] - gbs[43];
  K_c = exp(-dG);
  q_f =   S_tbc[271] * k_f * cs[1] * cs[43];
  q_b = - S_tbc[271] * k_f/K_c * cs[14] * cs[32];
  q[271] =   q_f + q_b;

  // Reaction #272
  k_f = exp(log(20000.000000000004)+2*logT-(1006.4390669975314/T));
   dG =  - gbs[44] + gbs[45];
  K_c = exp(-dG);
  q_f =   S_tbc[272] * k_f * cs[1] * cs[44];
  q_b = - S_tbc[272] * k_f/K_c * cs[1] * cs[45];
  q[272] =   q_f + q_b;

  // Reaction #273
  k_f = 9000000000.000002;
   dG =   gbs[14] - gbs[27] - gbs[35] + gbs[43];
  K_c = exp(-dG);
  q_f =   S_tbc[273] * k_f * cs[27] * cs[35];
  q_b = - S_tbc[273] * k_f/K_c * cs[14] * cs[43];
  q[273] =   q_f + q_b;

  // Reaction #274
  k_f = exp(log(610000000000.0001)-0.31*logT-(145.93366471464208/T));
   dG =   gbs[1] - gbs[12] - gbs[30] + gbs[41];
  K_c = exp(-dG);
  q_f =   S_tbc[274] * k_f * cs[12] * cs[30];
  q_b = - S_tbc[274] * k_f/K_c * cs[1] * cs[41];
  q[274] =   q_f + q_b;

  // Reaction #275
  k_f = exp(log(3700000000.0000005)+0.15*logT-(-45.28975801488892/T));
   dG =   gbs[0] - gbs[12] - gbs[30] + gbs[40];
  K_c = exp(-dG);
  q_f =   S_tbc[275] * k_f * cs[12] * cs[30];
  q_b = - S_tbc[275] * k_f/K_c * cs[0] * cs[40];
  q[275] =   q_f + q_b;

  // Reaction #276
  k_f = exp(log(540.0)+2.4*logT-(4989.421674640263/T));
   dG =   gbs[0] - gbs[1] + gbs[32] - gbs[33];
  K_c = exp(-dG);
  q_f =   S_tbc[276] * k_f * cs[1] * cs[33];
  q_b = - S_tbc[276] * k_f/K_c * cs[0] * cs[32];
  q[276] =   q_f + q_b;

  // Reaction #277
  k_f = exp(log(50000.00000000001)+1.6*logT-(480.57465449132127/T));
   dG =  - gbs[4] + gbs[5] + gbs[32] - gbs[33];
  K_c = exp(-dG);
  q_f =   S_tbc[277] * k_f * cs[4] * cs[33];
  q_b = - S_tbc[277] * k_f/K_c * cs[5] * cs[32];
  q[277] =   q_f + q_b;

  // Reaction #278
  k_f = exp(log(9400.000000000002)+1.94*logT-(3250.7981864020267/T));
   dG =  - gbs[2] + gbs[4] + gbs[32] - gbs[33];
  K_c = exp(-dG);
  q_f =   S_tbc[278] * k_f * cs[2] * cs[33];
  q_b = - S_tbc[278] * k_f/K_c * cs[4] * cs[32];
  q[278] =   q_f + q_b;

  // Reaction #279
  k_f = exp(log(10000000000.000002)-(7221.200305707288/T));
   dG =   gbs[14] - gbs[15] - gbs[31] + gbs[38];
  K_c = exp(-dG);
  q_f =   S_tbc[279] * k_f * cs[15] * cs[31];
  q_b = - S_tbc[279] * k_f/K_c * cs[14] * cs[38];
  q[279] =   q_f + q_b;

  // Reaction #280
  k_f = exp(log(6160000000000.001)-0.752*logT-(173.61073905707417/T));
   dG =   gbs[35] - gbs[36] - gbs[39] + gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[280] * k_f * cs[36] * cs[39];
  q_b = - S_tbc[280] * k_f/K_c * cs[35] * cs[46];
  q[280] =   q_f + q_b;

  // Reaction #281
  k_f = exp(log(3250000000.0000005)-(-354.76977111662984/T));
   dG =   gbs[15] - gbs[36] + gbs[37] - gbs[46];
  K_c = exp(-dG);
  q_f =   S_tbc[281] * k_f * cs[36] * cs[46];
  q_b = - S_tbc[281] * k_f/K_c * cs[15] * cs[37];
  q[281] =   q_f + q_b;

  // Reaction #282
  k_f = exp(log(3000000000.0000005)-(5686.380728536053/T));
   dG =   gbs[14] - gbs[15] - gbs[30] + gbs[35];
  K_c = exp(-dG);
  q_f =   S_tbc[282] * k_f * cs[15] * cs[30];
  q_b = - S_tbc[282] * k_f/K_c * cs[14] * cs[35];
  q[282] =   q_f + q_b;

  // Reaction #283
  k_f = 33700000000.000008;
   dG =   gbs[0] + gbs[1] - gbs[2] - gbs[12] + gbs[14];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[283] * k_f * cs[2] * cs[12];
  q_b = - S_tbc[283] * k_f/K_c * cs[0] * cs[1] * cs[14];
  q[283] =   q_f;

  // Reaction #284
  k_f = exp(log(6700.000000000001)+1.83*logT-(110.70829736972846/T));
   dG =   gbs[1] - gbs[2] - gbs[24] + gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[284] * k_f * cs[2] * cs[24];
  q_b = - S_tbc[284] * k_f/K_c * cs[1] * cs[51];
  q[284] =   q_f + q_b;

  // Reaction #285
  k_f = 109600000000.00002;
   dG =   gbs[1] - gbs[2] - gbs[25] + gbs[52];
  K_c = exp(-dG);
  q_f =   S_tbc[285] * k_f * cs[2] * cs[25];
  q_b = - S_tbc[285] * k_f/K_c * cs[1] * cs[52];
  q[285] =   q_f + q_b;

  // Reaction #286
  k_f = exp(log(5000000000000.001)-(8720.794515533611/T));
   dG =   gbs[3] - gbs[4] + gbs[5] - gbs[6];
  K_c = exp(-dG);
  q_f =   S_tbc[286] * k_f * cs[4] * cs[6];
  q_b = - S_tbc[286] * k_f/K_c * cs[3] * cs[5];
  q[286] =   q_f + q_b;

  // Reaction #287
  k_f = exp(log(8000000.000000001)+0.5*logT-(-883.1502812903339/T));
   dG =   gbs[0] - gbs[4] - gbs[12] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[287] * k_f * cs[4] * cs[12];
  q_b = - S_tbc[287] * k_f/K_c * cs[0] * cs[17];
  q[287] =   q_f;

  // Reaction #288
  k_f = exp(log(1970000000.0000002)+0.43*logT-(-186.19122739454332/T));
   dG =  - gbs[0] - gbs[9] + gbs[12];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #288
  Fcent = (1.0 - (0.578))*exp(-T/(122.0)) + (0.578) *exp(-T/(2535.0)) + exp(-(9365.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(4.820000000000001e+19)-2.8*logT-(296.8995247642718/T));
  Pr = S_tbc[288]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[0] * cs[9];
  q_b = - k_f/K_c * cs[12];
  q[288] =   q_f + q_b;

  // Reaction #289
  k_f = exp(log(5800000000.000001)-(754.8293002481486/T));
   dG =  2.0*gbs[1] - gbs[3] - gbs[10] + gbs[15];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[289] * k_f * cs[3] * cs[10];
  q_b = - S_tbc[289] * k_f/K_c * pow(cs[1],2.0) * cs[15];
  q[289] =   q_f;

  // Reaction #290
  k_f = exp(log(2400000000.0000005)-(754.8293002481486/T));
   dG =   gbs[2] - gbs[3] - gbs[10] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[290] * k_f * cs[3] * cs[10];
  q_b = - S_tbc[290] * k_f/K_c * cs[2] * cs[17];
  q[290] =   q_f + q_b;

  // Reaction #291
  k_f = exp(log(200000000000.00003)-(5529.879453617937/T));
   dG =  2.0*gbs[1] -2.0*gbs[10] + gbs[22];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[291] * k_f * pow(cs[10],2.0);
  q_b = - S_tbc[291] * k_f/K_c * pow(cs[1],2.0) * cs[22];
  q[291] =   q_f;

  // Reaction #292
  k_f = exp(log(68200000.00000001)+0.25*logT-(-470.510263821346/T));
   dG =   gbs[0] - gbs[5] - gbs[11] + gbs[17];
  K_c = exp(-dG);
  q_f =   S_tbc[292] * k_f * cs[5] * cs[11];
  q_b = - S_tbc[292] * k_f/K_c * cs[0] * cs[17];
  q[292] =   q_f;

  // Reaction #293
  k_f = exp(log(303000000.00000006)+0.29*logT-(5.5354148684864235/T));
   dG =   gbs[2] - gbs[3] - gbs[23] + gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[293] * k_f * cs[3] * cs[23];
  q_b = - S_tbc[293] * k_f/K_c * cs[2] * cs[51];
  q[293] =   q_f + q_b;

  // Reaction #294
  k_f = exp(log(1337.0000000000002)+1.61*logT-(-193.23630086352605/T));
   dG =  - gbs[3] + gbs[6] + gbs[22] - gbs[23];
  K_c = exp(-dG);
  q_f =   S_tbc[294] * k_f * cs[3] * cs[23];
  q_b = - S_tbc[294] * k_f/K_c * cs[6] * cs[22];
  q[294] =   q_f + q_b;

  // Reaction #295
  k_f = exp(log(2920000000.0000005)-(909.8209165657685/T));
   dG =  - gbs[2] + gbs[4] + gbs[51] - gbs[52];
  K_c = exp(-dG);
  q_f =   S_tbc[295] * k_f * cs[2] * cs[52];
  q_b = - S_tbc[295] * k_f/K_c * cs[4] * cs[51];
  q[295] =   q_f + q_b;

  // Reaction #296
  k_f = exp(log(2920000000.0000005)-(909.8209165657685/T));
   dG =  - gbs[2] + gbs[4] + gbs[12] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[296] * k_f * cs[2] * cs[52];
  q_b = - S_tbc[296] * k_f/K_c * cs[4] * cs[12] * cs[14];
  q[296] =   q_f;

  // Reaction #297
  k_f = exp(log(30100000000.000004)-(19701.044736476677/T));
   dG =  - gbs[3] + gbs[6] + gbs[12] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[297] * k_f * cs[3] * cs[52];
  q_b = - S_tbc[297] * k_f/K_c * cs[6] * cs[12] * cs[14];
  q[297] =   q_f;

  // Reaction #298
  k_f = exp(log(2050000.0000000005)+1.16*logT-(1210.2429780645316/T));
   dG =   gbs[0] - gbs[1] + gbs[51] - gbs[52];
  K_c = exp(-dG);
  q_f =   S_tbc[298] * k_f * cs[1] * cs[52];
  q_b = - S_tbc[298] * k_f/K_c * cs[0] * cs[51];
  q[298] =   q_f + q_b;

  // Reaction #299
  k_f = exp(log(2050000.0000000005)+1.16*logT-(1210.2429780645316/T));
   dG =   gbs[0] - gbs[1] + gbs[12] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[299] * k_f * cs[1] * cs[52];
  q_b = - S_tbc[299] * k_f/K_c * cs[0] * cs[12] * cs[14];
  q[299] =   q_f;

  // Reaction #300
  k_f = exp(log(23430000.000000004)+0.73*logT-(-560.0833407841262/T));
   dG =  - gbs[4] + gbs[5] + gbs[12] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[300] * k_f * cs[4] * cs[52];
  q_b = - S_tbc[300] * k_f/K_c * cs[5] * cs[12] * cs[14];
  q[300] =   q_f;

  // Reaction #301
  k_f = exp(log(3010000000.0000005)-(5999.8864979057835/T));
   dG =  - gbs[6] + gbs[7] + gbs[12] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[301] * k_f * cs[6] * cs[52];
  q_b = - S_tbc[301] * k_f/K_c * cs[7] * cs[12] * cs[14];
  q[301] =   q_f;

  // Reaction #302
  k_f = exp(log(2720.0000000000005)+1.77*logT-(2979.059638312693/T));
   dG =   gbs[13] + gbs[14] - gbs[52];
  K_c = prefRuT*exp(-dG);
  //  Three Body Reaction #302
  q_f =   S_tbc[302] * k_f * cs[52];
  q_b = - S_tbc[302] * k_f/K_c * cs[13] * cs[14];
  q[302] =   q_f;

  // Reaction #303
  k_f = exp(log(486500000.00000006)+0.422*logT-(-883.1502812903339/T));
   dG =  - gbs[1] - gbs[28] + gbs[51];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #303
  Fcent = (1.0 - (0.465))*exp(-T/(201.0)) + (0.465) *exp(-T/(1772.9999999999998)) + exp(-(5333.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(1.0120000000000002e+36)-7.63*logT-(1939.4080821042432/T));
  Pr = S_tbc[303]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[28];
  q_b = - k_f/K_c * cs[51];
  q[303] =   q_f + q_b;

  // Reaction #304
  k_f = 150000000000.00003;
   dG =   gbs[1] - gbs[2] + gbs[10] + gbs[15] - gbs[51];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[304] * k_f * cs[2] * cs[51];
  q_b = - S_tbc[304] * k_f/K_c * cs[1] * cs[10] * cs[15];
  q[304] =   q_f;

  // Reaction #305
  k_f = 18100000.000000004;
   dG =  - gbs[3] + gbs[4] + gbs[14] + gbs[17] - gbs[51];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[305] * k_f * cs[3] * cs[51];
  q_b = - S_tbc[305] * k_f/K_c * cs[4] * cs[14] * cs[17];
  q[305] =   q_f;

  // Reaction #306
  k_f = 23500000.000000004;
   dG =  - gbs[3] + gbs[4] +2.0*gbs[16] - gbs[51];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[306] * k_f * cs[3] * cs[51];
  q_b = - S_tbc[306] * k_f/K_c * cs[4] * pow(cs[16],2.0);
  q[306] =   q_f;

  // Reaction #307
  k_f = 22000000000.000004;
   dG =  - gbs[1] + gbs[12] + gbs[16] - gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[307] * k_f * cs[1] * cs[51];
  q_b = - S_tbc[307] * k_f/K_c * cs[12] * cs[16];
  q[307] =   q_f + q_b;

  // Reaction #308
  k_f = 11000000000.000002;
   dG =   gbs[0] - gbs[1] + gbs[28] - gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[308] * k_f * cs[1] * cs[51];
  q_b = - S_tbc[308] * k_f/K_c * cs[0] * cs[28];
  q[308] =   q_f + q_b;

  // Reaction #309
  k_f = 12000000000.000002;
   dG =  - gbs[4] + gbs[5] + gbs[28] - gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[309] * k_f * cs[4] * cs[51];
  q_b = - S_tbc[309] * k_f/K_c * cs[5] * cs[28];
  q[309] =   q_f + q_b;

  // Reaction #310
  k_f = 30100000000.000004;
   dG =  - gbs[4] + gbs[16] + gbs[18] - gbs[51];
  K_c = exp(-dG);
  q_f =   S_tbc[310] * k_f * cs[4] * cs[51];
  q_b = - S_tbc[310] * k_f/K_c * cs[16] * cs[18];
  q[310] =   q_f + q_b;

  // Reaction #311
  k_f = 9430000000.000002;
   dG =  - gbs[12] - gbs[25] + gbs[50];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #311
  Fcent = (1.0 - (0.1527))*exp(-T/(291.0)) + (0.1527) *exp(-T/(2742.0)) + exp(-(7748.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(2.7100000000000003e+68)-16.82*logT-(6574.563205161375/T));
  Pr = S_tbc[311]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[12] * cs[25];
  q_b = - k_f/K_c * cs[50];
  q[311] =   q_f + q_b;

  // Reaction #312
  k_f = exp(log(193.00000000000003)+2.68*logT-(1869.9637864814135/T));
   dG =  - gbs[2] + gbs[4] + gbs[49] - gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[312] * k_f * cs[2] * cs[50];
  q_b = - S_tbc[312] * k_f/K_c * cs[4] * cs[49];
  q[312] =   q_f + q_b;

  // Reaction #313
  k_f = exp(log(1320.0000000000002)+2.54*logT-(3399.7511683176613/T));
   dG =   gbs[0] - gbs[1] + gbs[49] - gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[313] * k_f * cs[1] * cs[50];
  q_b = - S_tbc[313] * k_f/K_c * cs[0] * cs[49];
  q[313] =   q_f + q_b;

  // Reaction #314
  k_f = exp(log(31600.000000000004)+1.8*logT-(470.0070442878472/T));
   dG =  - gbs[4] + gbs[5] + gbs[49] - gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[314] * k_f * cs[4] * cs[50];
  q_b = - S_tbc[314] * k_f/K_c * cs[5] * cs[49];
  q[314] =   q_f + q_b;

  // Reaction #315
  k_f = exp(log(0.37800000000000006)+2.72*logT-(754.8293002481486/T));
   dG =   gbs[6] - gbs[7] - gbs[49] + gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[315] * k_f * cs[7] * cs[49];
  q_b = - S_tbc[315] * k_f/K_c * cs[6] * cs[50];
  q[315] =   q_f + q_b;

  // Reaction #316
  k_f = exp(log(0.0009030000000000002)+3.65*logT-(3600.03254265017/T));
   dG =  - gbs[12] + gbs[13] + gbs[49] - gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[316] * k_f * cs[12] * cs[50];
  q_b = - S_tbc[316] * k_f/K_c * cs[13] * cs[49];
  q[316] =   q_f + q_b;

  // Reaction #317
  k_f = exp(log(2550.0000000000005)+1.6*logT-(2868.3513409429647/T));
   dG =  - gbs[12] - gbs[24] + gbs[49];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #317
  Fcent = (1.0 - (0.1894))*exp(-T/(277.0)) + (0.1894) *exp(-T/(8748.0)) + exp(-(7891.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(3.0000000000000007e+57)-14.6*logT-(9143.498923672574/T));
  Pr = S_tbc[317]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[12] * cs[24];
  q_b = - k_f/K_c * cs[49];
  q[317] =   q_f + q_b;

  // Reaction #318
  k_f = 96400000000.00002;
   dG =  - gbs[2] + gbs[17] + gbs[25] - gbs[49];
  K_c = exp(-dG);
  q_f =   S_tbc[318] * k_f * cs[2] * cs[49];
  q_b = - S_tbc[318] * k_f/K_c * cs[17] * cs[25];
  q[318] =   q_f + q_b;

  // Reaction #319
  k_f = 36130000000.00001;
   dG =  - gbs[1] - gbs[49] + gbs[50];
  K_c = exp(-dG)/prefRuT;
  //  Troe Reaction #319
  Fcent = (1.0 - (0.315))*exp(-T/(369.0)) + (0.315) *exp(-T/(3284.9999999999995)) + exp(-(6667.0)/T);
  C = - 0.4 - 0.67*log10(Fcent);
  N =   0.75 - 1.27*log10(Fcent);
  k0 = exp(log(4.420000000000001e+55)-13.545*logT-(5715.0642419454825/T));
  Pr = S_tbc[319]*k0/k_f;
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent)/(1.0+f1*f1));
  pmod = Pr/(1.0 + Pr) * F_pdr;
  k_f = k_f*pmod;
  q_f =   k_f * cs[1] * cs[49];
  q_b = - k_f/K_c * cs[50];
  q[319] =   q_f + q_b;

  // Reaction #320
  k_f = exp(log(4060.0000000000005)+2.19*logT-(447.8653848139015/T));
   dG =  - gbs[1] + gbs[12] + gbs[25] - gbs[49];
  K_c = exp(-dG);
  q_f =   S_tbc[320] * k_f * cs[1] * cs[49];
  q_b = - S_tbc[320] * k_f/K_c * cs[12] * cs[25];
  q[320] =   q_f + q_b;

  // Reaction #321
  k_f = 24100000000.000004;
   dG =  - gbs[4] + gbs[18] + gbs[25] - gbs[49];
  K_c = exp(-dG);
  q_f =   S_tbc[321] * k_f * cs[4] * cs[49];
  q_b = - S_tbc[321] * k_f/K_c * cs[18] * cs[25];
  q[321] =   q_f + q_b;

  // Reaction #322
  k_f = exp(log(25500000.000000004)+0.255*logT-(-474.5360200893361/T));
   dG =   gbs[3] - gbs[6] - gbs[49] + gbs[50];
  K_c = exp(-dG);
  q_f =   S_tbc[322] * k_f * cs[6] * cs[49];
  q_b = - S_tbc[322] * k_f/K_c * cs[3] * cs[50];
  q[322] =   q_f + q_b;

  // Reaction #323
  k_f = 24100000000.000004;
   dG =   gbs[4] - gbs[6] + gbs[17] + gbs[25] - gbs[49];
  K_c = prefRuT*exp(-dG);
  q_f =   S_tbc[323] * k_f * cs[6] * cs[49];
  q_b = - S_tbc[323] * k_f/K_c * cs[4] * cs[17] * cs[25];
  q[323] =   q_f;

  // Reaction #324
  k_f = exp(log(19270000000.000004)-0.32*logT);
   dG =  - gbs[12] +2.0*gbs[25] - gbs[49];
  K_c = exp(-dG);
  q_f =   S_tbc[324] * k_f * cs[12] * cs[49];
  q_b = - S_tbc[324] * k_f/K_c * pow(cs[25],2.0);
  q[324] =   q_f + q_b;

  // ----------------------------------------------------------- >
  // Source terms. --------------------------------------------- >
  // ----------------------------------------------------------- >

  b.omega(i,j,k,1) = th.MW(0) * ( -q[2] +q[7] +q[38] +q[39] +q[40] +q[41] +q[44] +q[46] +q[48] +q[50] +q[52] +q[54] +q[57] +q[59] +q[64] +q[67] +q[68] +q[72] +q[74] +q[76] +q[77] +q[79] -q[82] -q[83] -q[125] -q[135] +q[136] -q[145] -q[171] +q[173] +q[190] +q[196] +q[201] +q[208] +q[213] -q[220] +q[265] +q[275] +q[276] +q[283] +q[287] -q[288] +q[292] +q[298] +q[299] +q[308] +q[313]);
  b.omega(i,j,k,2) = th.MW(1) * ( -q[1] +q[2] +q[5] +q[6] +q[8] +q[9] +q[13] +q[20] +q[23] +q[27] -q[32] -q[33] -q[34] -q[35] -q[36] -q[37] -2.0*q[38] -2.0*q[39] -2.0*q[40] -2.0*q[41] -q[42] -q[43] -q[44] -q[45] -q[46] -q[47] -q[48] -q[49] -q[50] -q[51] -q[52] -q[53] -q[54] -q[55] -q[56] -q[57] -q[58] -q[59] -q[60] -q[61] -q[62] -q[64] -q[65] -q[66] -q[67] -q[68] -q[69] -q[70] -q[71] -q[72] -q[73] -q[74] -q[75] -q[76] -q[77] -q[78] -q[79] -q[80] +q[83] +q[89] +q[90] +q[91] +q[93] +q[98] +q[105] +q[106] +q[107] +q[122] +q[123] +q[125] +q[126] +q[127] +q[128] +q[129] +q[132] +q[134] +q[135] +q[137] +q[143] +q[145] +q[148] +q[158] +q[165] +q[166] +q[171] +q[179] -q[182] -q[188] +q[189] -q[190] +q[191] +q[195] +q[198] +q[200] -q[201] +q[203] +q[204] -q[208] -q[211] -q[213] +q[217] +q[220] -q[222] +q[223] +q[229] +q[230] +q[233] +q[234] -q[236] +q[246] +q[248] +q[250] +q[251] +q[253] +q[256] +q[259] -q[260] -q[264] -q[265] -q[270] -q[271] +q[274] -q[276] +q[283] +q[284] +q[285] +2.0*q[289] +2.0*q[291] -q[298] -q[299] -q[303] +q[304] -q[307] -q[308] -q[313] -q[319] -q[320]);
  b.omega(i,j,k,3) = th.MW(2) * ( -2.0*q[0] -q[1] -q[2] -q[3] -q[4] -q[5] -q[6] -q[7] -q[8] -q[9] -q[10] -q[11] -q[12] -q[13] -q[14] -q[15] -q[16] -q[17] -q[18] -q[19] -q[20] -q[21] -q[22] -q[23] -q[24] -q[25] -q[26] -q[27] -q[28] -q[29] +q[30] +q[37] +q[43] +q[85] +q[121] +q[124] +q[154] +q[177] +q[178] -q[180] -q[181] +q[184] -q[186] -q[187] -q[189] +q[193] -q[199] -q[200] -q[206] -q[207] -q[212] -q[216] +q[219] -q[221] -q[230] -q[231] -q[232] +q[243] +q[245] -q[256] -q[257] +q[258] -q[261] -q[262] -q[263] -q[278] -q[283] -q[284] -q[285] +q[290] +q[293] -q[295] -q[296] -q[304] -q[312] -q[318]);
  b.omega(i,j,k,4) = th.MW(3) * ( q[0] +q[3] -q[30] -q[31] -q[32] -q[33] -q[34] -q[35] -q[36] -q[37] +q[44] +q[86] +q[114] +q[115] +q[117] -q[121] -q[124] -q[134] -q[143] -q[144] -q[154] -q[155] -q[167] -q[168] -q[169] -q[170] -q[172] -q[174] -q[175] -q[178] +q[180] +q[187] -q[193] -q[194] -q[205] -q[215] -q[219] -q[225] -q[258] +q[286] -q[289] -q[290] -q[293] -q[294] -q[297] -q[305] -q[306] +q[322]);
  b.omega(i,j,k,5) = th.MW(4) * ( q[1] +q[2] +q[3] +q[4] +q[10] +q[12] +q[14] +q[15] +q[16] +q[17] +q[18] +q[21] +q[26] +q[28] +q[37] -q[42] +2.0*q[45] +q[47] +q[60] +q[65] -q[83] -2.0*q[84] -2.0*q[85] -q[86] -q[87] -q[88] -q[89] -q[90] -q[91] -q[92] -q[93] -q[94] -q[95] -q[96] -q[97] -q[98] -q[99] -q[100] -q[101] -q[102] -q[103] -q[104] -q[105] -q[106] -q[107] -q[108] -q[109] -q[110] -q[111] -q[112] -q[113] +q[116] +q[118] +q[119] +q[134] +q[143] +q[155] +q[175] -q[179] +q[182] -q[183] +q[185] +q[188] -q[191] -q[192] +q[194] +q[197] +q[199] -q[202] +q[206] -q[209] +q[212] -q[214] -q[217] +q[218] -q[223] +q[232] -q[233] -q[234] -q[235] +q[249] +q[252] +q[255] -q[259] +q[263] -q[266] -q[267] +q[270] -q[277] +q[278] -q[286] -q[287] +q[295] +q[296] -q[300] +q[305] +q[306] -q[309] -q[310] +q[312] -q[314] -q[321] +q[323]);
  b.omega(i,j,k,6) = th.MW(5) * ( q[42] +q[43] +q[47] +q[61] +q[66] +q[83] +q[85] +q[86] +q[87] +q[88] +q[92] +q[95] +q[96] +q[97] +q[99] +q[100] +q[101] +q[102] +q[103] +q[104] +q[108] +q[110] +q[111] +q[112] +q[113] -q[126] +q[144] -q[146] +q[192] -q[196] +q[202] +q[209] +q[214] -q[218] +q[254] +q[266] +q[277] +q[286] -q[292] +q[300] +q[309] +q[314]);
  b.omega(i,j,k,7) = th.MW(6) * ( -q[3] +q[4] +q[31] +q[32] +q[33] +q[34] +q[35] +q[36] -q[43] -q[44] -q[45] +q[46] -q[86] +q[87] +q[88] -2.0*q[114] -2.0*q[115] -q[116] -q[117] -q[118] -q[119] -q[120] +q[156] +q[167] +q[168] +q[169] +q[174] +q[183] -q[185] +q[205] +q[215] -q[286] +q[294] +q[297] -q[301] +q[315] -q[322] -q[323]);
  b.omega(i,j,k,8) = th.MW(7) * ( -q[4] -q[46] -q[47] +q[84] -q[87] -q[88] +q[114] +q[115] +q[120] -q[156] +q[301] -q[315]);
  b.omega(i,j,k,9) = th.MW(8) * ( q[48] -q[89] -q[121] -q[122] -q[123] -q[238] -q[243] -q[244]);
  b.omega(i,j,k,10) = th.MW(9) * ( -q[5] +q[19] -q[48] +q[50] -q[90] +q[92] -q[124] -q[125] -q[126] -q[127] -q[128] -q[129] -q[130] -q[131] -q[132] -q[133] -q[239] -q[240] -q[245] -q[246] -q[247] -q[288]);
  b.omega(i,j,k,11) = th.MW(10) * ( -q[6] +q[22] +q[29] -q[49] -q[91] -q[92] +q[95] -q[116] -q[122] +q[125] -q[127] -q[134] -q[135] -2.0*q[136] -q[137] -q[138] -q[139] -q[140] +q[141] +q[142] +q[147] +q[150] +q[151] +q[237] -q[241] -q[248] -q[249] -q[250] +q[260] -q[289] -q[290] -2.0*q[291] +q[304]);
  b.omega(i,j,k,12) = th.MW(11) * ( -q[7] -q[8] -q[50] +q[61] +q[66] +q[78] -q[93] +q[96] -q[141] -q[142] -q[143] -q[144] -q[145] -q[146] -q[147] -q[148] -q[149] -q[150] -q[151] -q[152] -q[153] -q[242] -q[251] -q[252] -q[253] -q[292]);
  b.omega(i,j,k,13) = th.MW(12) * ( -q[9] +q[10] +q[24] +q[25] +q[49] -q[51] +q[52] +q[60] +q[65] +q[80] -q[94] -q[95] -q[96] +q[97] +q[109] -q[117] -q[118] -q[123] -q[128] +q[135] -q[137] +2.0*q[138] +q[145] -q[148] +2.0*q[149] +q[153] -q[154] -q[155] -q[156] -2.0*q[157] -2.0*q[158] -q[159] -q[160] -q[161] -q[162] -q[163] -q[164] -q[210] -q[254] -q[255] -q[274] -q[275] -q[283] -q[287] +q[288] +q[296] +q[297] +q[299] +q[300] +q[301] +q[307] -q[311] -q[316] -q[317] +q[320] -q[324]);
  b.omega(i,j,k,14) = th.MW(13) * ( -q[10] +q[51] -q[52] -q[97] +q[117] -q[129] -q[138] -q[149] +q[156] +q[159] +q[160] +q[161] +q[162] +q[163] +q[164] +q[210] +q[302] +q[316]);
  b.omega(i,j,k,15) = th.MW(14) * ( q[5] +q[7] -q[11] +q[12] +q[19] +q[22] +2.0*q[27] -q[30] +q[54] +q[78] +q[80] -q[82] +q[89] -q[98] +q[99] +q[109] -q[119] +q[121] -q[130] +q[131] +q[133] +q[134] -q[139] +q[140] +q[143] +q[144] +q[152] +q[159] +q[165] +q[166] +q[167] +q[170] +2.0*q[175] +2.0*q[176] +q[216] +q[221] +q[222] +q[223] +q[224] +q[226] +q[227] +q[231] +q[235] +q[244] +q[256] +q[262] +q[264] +q[268] +q[271] +q[273] +q[279] +q[282] +q[283] +q[296] +q[297] +q[299] +q[300] +q[301] +q[302] +q[305]);
  b.omega(i,j,k,16) = th.MW(15) * ( q[11] +q[13] +q[29] +q[30] +q[98] +q[119] -q[131] -q[152] +q[225] +q[228] +q[261] +q[267] -q[279] +q[281] -q[282] +q[289] +q[304]);
  b.omega(i,j,k,17) = th.MW(16) * ( q[6] +q[8] -q[12] -q[13] +q[14] +q[24] +q[31] -q[53] -q[54] +q[57] +q[90] -q[99] +q[100] +q[120] +q[124] +q[131] -q[159] +q[160] -q[165] -q[166] -q[167] +q[170] +q[172] +q[247] +q[258] +q[259] +2.0*q[306] +q[307] +q[310]);
  b.omega(i,j,k,18) = th.MW(17) * ( q[9] -q[14] +q[15] +q[16] +q[25] -q[31] +q[53] -q[55] -q[56] -q[57] +q[59] +q[64] +q[82] +q[91] +q[93] -q[100] +q[101] +q[102] +q[116] -q[120] +q[126] -q[132] +q[152] +q[155] -q[160] +q[168] +q[169] +q[172] +q[287] +q[290] +q[292] +q[305] +q[318] +q[323]);
  b.omega(i,j,k,19) = th.MW(18) * ( -q[15] +q[17] +q[55] -q[58] -q[59] -q[60] -q[61] +q[63] +q[67] -q[101] +q[103] +q[161] -q[168] +q[310] +q[321]);
  b.omega(i,j,k,20) = th.MW(19) * ( -q[16] +q[18] +q[56] -q[62] -q[63] -q[64] -q[65] -q[66] +q[68] -q[102] +q[104] +q[118] +q[154] +q[162] -q[169]);
  b.omega(i,j,k,21) = th.MW(20) * ( -q[17] -q[18] +q[58] +q[62] -q[67] -q[68] +q[94] -q[103] -q[104] +q[146] -q[161] -q[162]);
  b.omega(i,j,k,22) = th.MW(21) * ( -q[19] +q[21] -q[69] -q[105] +q[108] +q[122] -q[170] -q[171]);
  b.omega(i,j,k,23) = th.MW(22) * ( -q[20] -q[21] -q[22] +q[69] -q[70] +q[72] -q[106] -q[107] -q[108] -q[109] +q[110] +q[123] +q[127] +q[133] +q[136] +q[171] +q[173] +q[176] +q[291] +q[294]);
  b.omega(i,j,k,24) = th.MW(23) * ( -q[23] +q[70] -q[71] -q[72] +q[74] -q[110] +q[111] +q[128] +q[140] +q[163] -q[172] -q[293] -q[294]);
  b.omega(i,j,k,25) = th.MW(24) * ( -q[24] +q[71] -q[73] -q[74] +q[76] -q[111] +q[129] +q[137] +q[148] -q[163] -q[173] +q[174] -q[284] -q[317]);
  b.omega(i,j,k,26) = th.MW(25) * ( -q[25] +q[26] +q[73] -q[75] -q[76] +q[77] +q[112] +q[153] +q[158] +q[164] -q[174] -q[285] -q[311] +q[318] +q[320] +q[321] +q[323] +2.0*q[324]);
  b.omega(i,j,k,27) = th.MW(26) * ( -q[26] +q[75] -q[77] -q[112] -q[153] +q[157] -q[164]);
  b.omega(i,j,k,28) = th.MW(27) * ( q[20] -q[27] +q[28] -q[78] +q[79] +q[105] +q[113] +q[130] -q[133] -q[140] -q[175] -2.0*q[176] -q[273]);
  b.omega(i,j,k,29) = th.MW(28) * ( q[23] -q[28] -q[29] -q[79] -q[80] +q[81] +q[106] -q[113] +q[132] +q[139] -q[303] +q[308] +q[309]);
  b.omega(i,j,k,30) = th.MW(29) * ( -q[81] +q[107]);
  b.omega(i,j,k,31) = th.MW(30) * ( -q[177] -q[178] -q[179] +q[190] +q[192] -q[195] +q[216] -q[224] +q[226] -q[237] +q[238] +q[239] +q[244] +q[247] -q[274] -q[275] -q[282]);
  b.omega(i,j,k,32) = th.MW(31) * ( -q[189] -q[190] -q[191] -q[192] -q[193] -q[194] -q[195] -q[196] -q[197] -q[198] +q[199] +q[201] +q[202] +q[207] +q[222] +q[231] +q[241] +q[242] +q[261] +q[268] -q[279]);
  b.omega(i,j,k,33) = th.MW(32) * ( -q[199] -q[200] -q[201] -q[202] +q[235] +q[264] +q[267] +q[271] +q[276] +q[277] +q[278]);
  b.omega(i,j,k,34) = th.MW(33) * ( -q[276] -q[277] -q[278]);
  b.omega(i,j,k,35) = th.MW(34) * ( -q[203] -q[204] -q[205] -q[206] -q[207] -q[208] -q[209] -q[210]);
  b.omega(i,j,k,36) = th.MW(35) * ( -q[177] +q[178] +q[179] +2.0*q[181] -q[185] -q[186] +q[187] +q[188] +q[189] +q[194] -q[197] -q[198] +q[207] -q[211] +q[212] +q[213] +q[214] +q[215] +q[221] +q[223] +q[225] -q[227] -q[228] -q[243] -q[244] -q[245] -q[246] -q[247] -q[248] -q[249] -q[250] -q[251] -q[252] -q[253] -q[254] -q[255] +q[257] -q[273] +q[280] +q[282]);
  b.omega(i,j,k,37) = th.MW(36) * ( q[185] +q[186] -q[187] -q[188] -q[280] -q[281]);
  b.omega(i,j,k,38) = th.MW(37) * ( -q[180] -q[181] -q[182] -q[183] -q[184] +q[198] +q[227] +q[281]);
  b.omega(i,j,k,39) = th.MW(38) * ( q[191] +q[193] +q[196] +q[200] +q[211] -q[212] -q[213] -q[214] -q[215] +q[262] +q[279]);
  b.omega(i,j,k,40) = th.MW(39) * ( -q[216] -q[217] -q[218] -q[219] -q[220] +q[229] +q[232] +q[238] +q[243] -q[280]);
  b.omega(i,j,k,41) = th.MW(40) * ( q[218] +q[220] -q[229] -q[230] -q[231] -q[232] -q[233] -q[234] -q[235] -q[236] +q[239] +q[241] +q[242] +q[245] +q[249] +q[252] +q[254] +q[257] +q[270] +q[275]);
  b.omega(i,j,k,42) = th.MW(41) * ( q[236] -q[237] +q[255] +q[274]);
  b.omega(i,j,k,43) = th.MW(42) * ( q[240] -q[256] -q[257] -q[258] -q[259] -q[260]);
  b.omega(i,j,k,44) = th.MW(43) * ( q[250] +q[253] -q[269] -q[270] -q[271] +q[273]);
  b.omega(i,j,k,45) = th.MW(44) * ( q[233] -q[272]);
  b.omega(i,j,k,46) = th.MW(45) * ( q[234] +q[248] +q[251] -q[261] -q[262] -q[263] -q[264] -q[265] -q[266] -q[267] -q[268] +q[269] +q[272]);
  b.omega(i,j,k,47) = th.MW(46) * ( q[217] +q[219] -q[221] -q[222] -q[223] -q[224] -q[225] -q[226] -q[227] -q[228] +q[230] +q[246] +q[263] +q[265] +q[266] +q[280] -q[281]);
  b.omega(i,j,k,48) = th.MW(47) * ( q[177] +q[180] +q[182] +q[183] +q[184] +q[195] +q[197] +q[203] +q[204] +q[205] +q[206] +q[208] +q[209] +q[210] +q[224] +q[228] +q[237] -q[238] -q[239] -q[240] -q[241] -q[242] +q[256] +q[258] +q[259] +q[260]);
  b.omega(i,j,k,49) = th.MW(48) * (0.0);
  b.omega(i,j,k,50) = th.MW(49) * ( q[312] +q[313] +q[314] -q[315] +q[316] +q[317] -q[318] -q[319] -q[320] -q[321] -q[322] -q[323] -q[324]);
  b.omega(i,j,k,51) = th.MW(50) * ( q[311] -q[312] -q[313] -q[314] +q[315] -q[316] +q[319] +q[322]);
  b.omega(i,j,k,52) = th.MW(51) * ( q[284] +q[293] +q[295] +q[298] +q[303] -q[304] -q[305] -q[306] -q[307] -q[308] -q[309] -q[310]);
  b.omega(i,j,k,53) = th.MW(52) * ( q[285] -q[295] -q[296] -q[297] -q[298] -q[299] -q[300] -q[301] -q[302]);

  // Add source terms to RHS
  for (int n=0; n<52; n++)
  {
    b.dQ(i,j,k,5+n) += b.omega(i,j,k,n+1);
  }
  // Compute constant pressure dTdt dYdt (for implicit chem integration)
  double dTdt = 0.0;
  for (int n=0; n<=52; n++)
  {
    dTdt -= b.qh(i,j,k,5+n) * b.omega(i,j,k,n+1);
    b.omega(i,j,k,n+1) /= b.Q(i,j,k,0);
  }
  dTdt /= b.qh(i,j,k,1) * b.Q(i,j,k,0);
  b.omega(i,j,k,0) = dTdt;

  });
}
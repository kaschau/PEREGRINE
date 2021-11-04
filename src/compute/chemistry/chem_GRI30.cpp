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
// ========================================================== //

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>

void chem_GRI30(block_ b, thtrdat_ th, int face/*=0*/, int i/*=0*/, int j/*=0*/, int k/*=0*/) {

// --------------------------------------------------------------|
// cc range
// --------------------------------------------------------------|
  MDRange3 range = get_range3(b, face, i, j, k);
  Kokkos::Experimental::UniqueToken<exec_space> token;
  int numIds = token.size();
  const int ns=53;
  const int nr=325;
  twoDview Y("Y", ns, numIds);
  twoDview cs("cs", ns, numIds);
  twoDview gbs("gbs", ns, numIds);

  Kokkos::parallel_for("Compute chemical source terms",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  int id = token.acquire();

  double T,logT,prefRuT;
  double rho = b.Q(i,j,k,0);
  T = b.q(i,j,k,4);
  logT = log(T);
  prefRuT = 101325.0/(th.Ru*T);

  // Compute nth species Y
  Y(ns-1,id) = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y(n,id) = b.q(i,j,k,5+n);
    Y(ns-1,id) -= Y(n,id);
  }
  Y(ns-1,id) = fmax(0.0,Y(ns-1,id));

  // Conecntrations
  for (int n=0; n<=ns-1; n++)
  {
    cs(n,id) = rho*Y(n,id)/th.MW[n];
  }

  // ----------------------------------------------------------- >
  // Chaperon efficiencies. ------------------------------------ >
  // ----------------------------------------------------------- >

  double S_tbc[325];
  for (int n = 0; n < 325; n++)
  {
     S_tbc[n] = 1.0;
  }

  S_tbc[0] = 2.4*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 15.4*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.75*cs(14,id) + 3.6*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.83*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[1] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[11] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + 6.0*cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 3.5*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.5*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[32] = cs(0,id) + cs(1,id) + cs(2,id) + cs(4,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + cs(13,id) + 0.75*cs(14,id) + 1.5*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 1.5*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[33] = cs(3,id);

  S_tbc[34] = cs(5,id);

  S_tbc[35] = cs(47,id);

  S_tbc[36] = cs(48,id);

  S_tbc[38] = cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + cs(14,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.63*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[39] = cs(0,id);

  S_tbc[40] = cs(5,id);

  S_tbc[41] = cs(15,id);

  S_tbc[42] = 0.73*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 3.65*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + cs(14,id) + cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.38*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[49] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[51] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 3.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[53] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[55] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[56] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[58] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[62] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[69] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[70] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[71] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[73] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[75] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[82] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[84] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[94] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[130] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[139] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[146] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[157] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[165] = cs(5,id);

  S_tbc[166] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[173] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[184] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.625*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[186] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[204] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[211] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[226] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[229] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[236] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[240] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[268] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[288] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[302] = cs(12,id);

  S_tbc[303] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[311] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[317] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  S_tbc[319] = 2.0*cs(0,id) + cs(1,id) + cs(2,id) + cs(3,id) + cs(4,id) + 6.0*cs(5,id) + cs(6,id) + cs(7,id) + cs(8,id) + cs(9,id) + cs(10,id) + cs(11,id) + cs(12,id) + 2.0*cs(13,id) + 1.5*cs(14,id) + 2.0*cs(15,id) + cs(16,id) + cs(17,id) + cs(18,id) + cs(19,id) + cs(20,id) + cs(21,id) + cs(22,id) + cs(23,id) + cs(24,id) + cs(25,id) + 3.0*cs(26,id) + cs(27,id) + cs(28,id) + cs(29,id) + cs(30,id) + cs(31,id) + cs(32,id) + cs(33,id) + cs(34,id) + cs(35,id) + cs(36,id) + cs(37,id) + cs(38,id) + cs(39,id) + cs(40,id) + cs(41,id) + cs(42,id) + cs(43,id) + cs(44,id) + cs(45,id) + cs(46,id) + cs(47,id) + 0.7*cs(48,id) + cs(49,id) + cs(50,id) + cs(51,id) + cs(52,id);

  // ----------------------------------------------------------- >
  // Gibbs energy. --------------------------------------------- >
  // ----------------------------------------------------------- >

  int m;
  double hi,scs;

  for (int n=0; n<=ns-1; n++)
  {
    m = ( T <= th.NASA7[n][0] ) ? 8 : 1;

    hi     = th.NASA7[n][m+0]                  +
             th.NASA7[n][m+1]*    T      / 2.0 +
             th.NASA7[n][m+2]*pow(T,2.0) / 3.0 +
             th.NASA7[n][m+3]*pow(T,3.0) / 4.0 +
             th.NASA7[n][m+4]*pow(T,4.0) / 5.0 +
             th.NASA7[n][m+5]/    T            ;
    scs    = th.NASA7[n][m+0]*log(T)           +
             th.NASA7[n][m+1]*    T            +
             th.NASA7[n][m+2]*pow(T,2.0) / 2.0 +
             th.NASA7[n][m+3]*pow(T,3.0) / 3.0 +
             th.NASA7[n][m+4]*pow(T,4.0) / 4.0 +
             th.NASA7[n][m+6]                  ;

    gbs(n,id) = hi-scs                         ;
  }

  // ----------------------------------------------------------- >
  // Rate Constants. ------------------------------------------- >
  // ----------------------------------------------------------- >

  double q_f[325],k_f[325];
  double q_b[325];

  double dG[325],K_c[325],q[325]; 

  k_f[0] = exp(log(120000000000.00002)-1.0*logT);
   dG[0] =  -2.0*gbs(2,id) + gbs(3,id);
  K_c[0] = exp(-dG[0])/prefRuT;

  k_f[1] = exp(log(500000000000.0001)-1.0*logT);
   dG[1] =  - gbs(1,id) - gbs(2,id) + gbs(4,id);
  K_c[1] = exp(-dG[1])/prefRuT;

  k_f[2] = exp(log(38.7)+2.7*logT-(3150.1542797022735/T));
   dG[2] =  - gbs(0,id) + gbs(1,id) - gbs(2,id) + gbs(4,id);
  K_c[2] = exp(-dG[2]);

  k_f[3] = 20000000000.000004;
   dG[3] =  - gbs(2,id) + gbs(3,id) + gbs(4,id) - gbs(6,id);
  K_c[3] = exp(-dG[3]);

  k_f[4] = exp(log(9630.0)+2*logT-(2012.8781339950629/T));
   dG[4] =  - gbs(2,id) + gbs(4,id) + gbs(6,id) - gbs(7,id);
  K_c[4] = exp(-dG[4]);

  k_f[5] = 57000000000.00001;
   dG[5] =   gbs(1,id) - gbs(2,id) - gbs(9,id) + gbs(14,id);
  K_c[5] = exp(-dG[5]);

  k_f[6] = 80000000000.00002;
   dG[6] =   gbs(1,id) - gbs(2,id) - gbs(10,id) + gbs(16,id);
  K_c[6] = exp(-dG[6]);

  k_f[7] = 15000000000.000002;
   dG[7] =   gbs(0,id) - gbs(2,id) - gbs(11,id) + gbs(14,id);
  K_c[7] = exp(-dG[7]);

  k_f[8] = 15000000000.000002;
   dG[8] =   gbs(1,id) - gbs(2,id) - gbs(11,id) + gbs(16,id);
  K_c[8] = exp(-dG[8]);

  k_f[9] = 50600000000.00001;
   dG[9] =   gbs(1,id) - gbs(2,id) - gbs(12,id) + gbs(17,id);
  K_c[9] = exp(-dG[9]);

  k_f[10] = exp(log(1020000.0000000001)+1.5*logT-(4327.687988089386/T));
   dG[10] =  - gbs(2,id) + gbs(4,id) + gbs(12,id) - gbs(13,id);
  K_c[10] = exp(-dG[10]);

  k_f[11] = exp(log(18000000.000000004)-(1200.1785873945562/T));
   dG[11] =  - gbs(2,id) - gbs(14,id) + gbs(15,id);
  K_c[11] = exp(-dG[11])/prefRuT;

  k_f[12] = 30000000000.000004;
   dG[12] =  - gbs(2,id) + gbs(4,id) + gbs(14,id) - gbs(16,id);
  K_c[12] = exp(-dG[12]);

  k_f[13] = 30000000000.000004;
   dG[13] =   gbs(1,id) - gbs(2,id) + gbs(15,id) - gbs(16,id);
  K_c[13] = exp(-dG[13]);

  k_f[14] = exp(log(39000000000.00001)-(1781.3971485856307/T));
   dG[14] =  - gbs(2,id) + gbs(4,id) + gbs(16,id) - gbs(17,id);
  K_c[14] = exp(-dG[14]);

  k_f[15] = 10000000000.000002;
   dG[15] =  - gbs(2,id) + gbs(4,id) + gbs(17,id) - gbs(18,id);
  K_c[15] = exp(-dG[15]);

  k_f[16] = 10000000000.000002;
   dG[16] =  - gbs(2,id) + gbs(4,id) + gbs(17,id) - gbs(19,id);
  K_c[16] = exp(-dG[16]);

  k_f[17] = exp(log(388.00000000000006)+2.5*logT-(1559.9805538461737/T));
   dG[17] =  - gbs(2,id) + gbs(4,id) + gbs(18,id) - gbs(20,id);
  K_c[17] = exp(-dG[17]);

  k_f[18] = exp(log(130.00000000000003)+2.5*logT-(2516.097667493829/T));
   dG[18] =  - gbs(2,id) + gbs(4,id) + gbs(19,id) - gbs(20,id);
  K_c[18] = exp(-dG[18]);

  k_f[19] = 50000000000.00001;
   dG[19] =  - gbs(2,id) + gbs(9,id) + gbs(14,id) - gbs(21,id);
  K_c[19] = exp(-dG[19]);

  k_f[20] = exp(log(13500.000000000002)+2*logT-(956.117113647655/T));
   dG[20] =   gbs(1,id) - gbs(2,id) - gbs(22,id) + gbs(27,id);
  K_c[20] = exp(-dG[20]);

  k_f[21] = exp(log(4.600000000000001e+16)-1.41*logT-(14568.205494789268/T));
   dG[21] =  - gbs(2,id) + gbs(4,id) + gbs(21,id) - gbs(22,id);
  K_c[21] = exp(-dG[21]);

  k_f[22] = exp(log(6940.000000000001)+2*logT-(956.117113647655/T));
   dG[22] =  - gbs(2,id) + gbs(10,id) + gbs(14,id) - gbs(22,id);
  K_c[22] = exp(-dG[22]);

  k_f[23] = 30000000000.000004;
   dG[23] =   gbs(1,id) - gbs(2,id) - gbs(23,id) + gbs(28,id);
  K_c[23] = exp(-dG[23]);

  k_f[24] = exp(log(12500.000000000002)+1.83*logT-(110.70829736972846/T));
   dG[24] =  - gbs(2,id) + gbs(12,id) + gbs(16,id) - gbs(24,id);
  K_c[24] = exp(-dG[24]);

  k_f[25] = 22400000000.000004;
   dG[25] =  - gbs(2,id) + gbs(12,id) + gbs(17,id) - gbs(25,id);
  K_c[25] = exp(-dG[25]);

  k_f[26] = exp(log(89800.00000000001)+1.92*logT-(2863.319145607977/T));
   dG[26] =  - gbs(2,id) + gbs(4,id) + gbs(25,id) - gbs(26,id);
  K_c[26] = exp(-dG[26]);

  k_f[27] = 100000000000.00002;
   dG[27] =   gbs(1,id) - gbs(2,id) +2.0*gbs(14,id) - gbs(27,id);
  K_c[27] = prefRuT*exp(-dG[27]);

  k_f[28] = exp(log(10000000000.000002)-(4025.7562679901257/T));
   dG[28] =  - gbs(2,id) + gbs(4,id) + gbs(27,id) - gbs(28,id);
  K_c[28] = exp(-dG[28]);

  k_f[29] = exp(log(1750000000.0000002)-(679.3463702233338/T));
   dG[29] =  - gbs(2,id) + gbs(10,id) + gbs(15,id) - gbs(28,id);
  K_c[29] = exp(-dG[29]);

  k_f[30] = exp(log(2500000000.0000005)-(24053.893701241002/T));
   dG[30] =   gbs(2,id) - gbs(3,id) - gbs(14,id) + gbs(15,id);
  K_c[30] = exp(-dG[30]);

  k_f[31] = exp(log(100000000000.00002)-(20128.78133995063/T));
   dG[31] =  - gbs(3,id) + gbs(6,id) + gbs(16,id) - gbs(17,id);
  K_c[31] = exp(-dG[31]);

  k_f[32] = exp(log(2800000000000.0005)-0.86*logT);
   dG[32] =  - gbs(1,id) - gbs(3,id) + gbs(6,id);
  K_c[32] = exp(-dG[32])/prefRuT;

  k_f[33] = exp(log(20800000000000.004)-1.24*logT);
   dG[33] =  - gbs(1,id) - gbs(3,id) + gbs(6,id);
  K_c[33] = exp(-dG[33])/prefRuT;

  k_f[34] = exp(log(11260000000000.002)-0.76*logT);
   dG[34] =  - gbs(1,id) - gbs(3,id) + gbs(6,id);
  K_c[34] = exp(-dG[34])/prefRuT;

  k_f[35] = exp(log(26000000000000.004)-1.24*logT);
   dG[35] =  - gbs(1,id) - gbs(3,id) + gbs(6,id);
  K_c[35] = exp(-dG[35])/prefRuT;

  k_f[36] = exp(log(700000000000.0001)-0.8*logT);
   dG[36] =  - gbs(1,id) - gbs(3,id) + gbs(6,id);
  K_c[36] = exp(-dG[36])/prefRuT;

  k_f[37] = exp(log(26500000000000.004)-0.6707*logT-(8575.364070352467/T));
   dG[37] =  - gbs(1,id) + gbs(2,id) - gbs(3,id) + gbs(4,id);
  K_c[37] = exp(-dG[37]);

  k_f[38] = exp(log(1000000000000.0002)-1.0*logT);
   dG[38] =   gbs(0,id) -2.0*gbs(1,id);
  K_c[38] = exp(-dG[38])/prefRuT;

  k_f[39] = exp(log(90000000000.00002)-0.6*logT);
   dG[39] =   gbs(0,id) -2.0*gbs(1,id);
  K_c[39] = exp(-dG[39])/prefRuT;

  k_f[40] = exp(log(60000000000000.01)-1.25*logT);
   dG[40] =   gbs(0,id) -2.0*gbs(1,id);
  K_c[40] = exp(-dG[40])/prefRuT;

  k_f[41] = exp(log(550000000000000.1)-2.0*logT);
   dG[41] =   gbs(0,id) -2.0*gbs(1,id);
  K_c[41] = exp(-dG[41])/prefRuT;

  k_f[42] = exp(log(2.2000000000000004e+16)-2.0*logT);
   dG[42] =  - gbs(1,id) - gbs(4,id) + gbs(5,id);
  K_c[42] = exp(-dG[42])/prefRuT;

  k_f[43] = exp(log(3970000000.0000005)-(337.66030697767184/T));
   dG[43] =  - gbs(1,id) + gbs(2,id) + gbs(5,id) - gbs(6,id);
  K_c[43] = exp(-dG[43]);

  k_f[44] = exp(log(44800000000.00001)-(537.4384617766818/T));
   dG[44] =   gbs(0,id) - gbs(1,id) + gbs(3,id) - gbs(6,id);
  K_c[44] = exp(-dG[44]);

  k_f[45] = exp(log(84000000000.00002)-(319.54440377171625/T));
   dG[45] =  - gbs(1,id) +2.0*gbs(4,id) - gbs(6,id);
  K_c[45] = exp(-dG[45]);

  k_f[46] = exp(log(12100.000000000002)+2*logT-(2616.741574193582/T));
   dG[46] =   gbs(0,id) - gbs(1,id) + gbs(6,id) - gbs(7,id);
  K_c[46] = exp(-dG[46]);

  k_f[47] = exp(log(10000000000.000002)-(1811.5903205955567/T));
   dG[47] =  - gbs(1,id) + gbs(4,id) + gbs(5,id) - gbs(7,id);
  K_c[47] = exp(-dG[47]);

  k_f[48] = 165000000000.00003;
   dG[48] =   gbs(0,id) - gbs(1,id) + gbs(8,id) - gbs(9,id);
  K_c[48] = exp(-dG[48]);

  k_f[49] = 600000000000.0001;
   dG[49] =  - gbs(1,id) - gbs(10,id) + gbs(12,id);
  K_c[49] = exp(-dG[49])/prefRuT;

  k_f[50] = 30000000000.000004;
   dG[50] =   gbs(0,id) - gbs(1,id) + gbs(9,id) - gbs(11,id);
  K_c[50] = exp(-dG[50]);

  k_f[51] = exp(log(13900000000000.002)-0.534*logT-(269.72566995533845/T));
   dG[51] =  - gbs(1,id) - gbs(12,id) + gbs(13,id);
  K_c[51] = exp(-dG[51])/prefRuT;

  k_f[52] = exp(log(660000.0000000001)+1.62*logT-(5454.899743126621/T));
   dG[52] =   gbs(0,id) - gbs(1,id) + gbs(12,id) - gbs(13,id);
  K_c[52] = exp(-dG[52]);

  k_f[53] = exp(log(1090000000.0000002)+0.48*logT-(-130.8370787096791/T));
   dG[53] =  - gbs(1,id) - gbs(16,id) + gbs(17,id);
  K_c[53] = exp(-dG[53])/prefRuT;

  k_f[54] = 73400000000.00002;
   dG[54] =   gbs(0,id) - gbs(1,id) + gbs(14,id) - gbs(16,id);
  K_c[54] = exp(-dG[54]);

  k_f[55] = exp(log(540000000.0000001)+0.454*logT-(1811.5903205955567/T));
   dG[55] =  - gbs(1,id) - gbs(17,id) + gbs(18,id);
  K_c[55] = exp(-dG[55])/prefRuT;

  k_f[56] = exp(log(540000000.0000001)+0.454*logT-(1308.370787096791/T));
   dG[56] =  - gbs(1,id) - gbs(17,id) + gbs(19,id);
  K_c[56] = exp(-dG[56])/prefRuT;

  k_f[57] = exp(log(57400.000000000015)+1.9*logT-(1379.8279608536157/T));
   dG[57] =   gbs(0,id) - gbs(1,id) + gbs(16,id) - gbs(17,id);
  K_c[57] = exp(-dG[57]);

  k_f[58] = exp(log(1055000000.0000002)+0.5*logT-(43.27687988089385/T));
   dG[58] =  - gbs(1,id) - gbs(18,id) + gbs(20,id);
  K_c[58] = exp(-dG[58])/prefRuT;

  k_f[59] = 20000000000.000004;
   dG[59] =   gbs(0,id) - gbs(1,id) + gbs(17,id) - gbs(18,id);
  K_c[59] = exp(-dG[59]);

  k_f[60] = exp(log(165000000.00000003)+0.65*logT-(-142.91434751364946/T));
   dG[60] =  - gbs(1,id) + gbs(4,id) + gbs(12,id) - gbs(18,id);
  K_c[60] = exp(-dG[60]);

  k_f[61] = exp(log(32800000000.000004)-0.09*logT-(306.9639154342471/T));
   dG[61] =  - gbs(1,id) + gbs(5,id) + gbs(11,id) - gbs(18,id);
  K_c[61] = exp(-dG[61]);

  k_f[62] = exp(log(2430000000.0000005)+0.515*logT-(25.160976674938286/T));
   dG[62] =  - gbs(1,id) - gbs(19,id) + gbs(20,id);
  K_c[62] = exp(-dG[62])/prefRuT;

  k_f[63] = exp(log(41500.00000000001)+1.63*logT-(968.1943824516253/T));
   dG[63] =   gbs(18,id) - gbs(19,id);
  K_c[63] = exp(-dG[63]);

  k_f[64] = 20000000000.000004;
   dG[64] =   gbs(0,id) - gbs(1,id) + gbs(17,id) - gbs(19,id);
  K_c[64] = exp(-dG[64]);

  k_f[65] = exp(log(1500000000.0000002)+0.5*logT-(-55.35414868486423/T));
   dG[65] =  - gbs(1,id) + gbs(4,id) + gbs(12,id) - gbs(19,id);
  K_c[65] = exp(-dG[65]);

  k_f[66] = exp(log(262000000000.00003)-0.23*logT-(538.4449008436793/T));
   dG[66] =  - gbs(1,id) + gbs(5,id) + gbs(11,id) - gbs(19,id);
  K_c[66] = exp(-dG[66]);

  k_f[67] = exp(log(17000.000000000004)+2.1*logT-(2450.679128138989/T));
   dG[67] =   gbs(0,id) - gbs(1,id) + gbs(18,id) - gbs(20,id);
  K_c[67] = exp(-dG[67]);

  k_f[68] = exp(log(4200.000000000001)+2.1*logT-(2450.679128138989/T));
   dG[68] =   gbs(0,id) - gbs(1,id) + gbs(19,id) - gbs(20,id);
  K_c[68] = exp(-dG[68]);

  k_f[69] = exp(log(100000000000000.02)-1.0*logT);
   dG[69] =  - gbs(1,id) - gbs(21,id) + gbs(22,id);
  K_c[69] = exp(-dG[69])/prefRuT;

  k_f[70] = exp(log(5600000000.000001)-(1207.7268803970378/T));
   dG[70] =  - gbs(1,id) - gbs(22,id) + gbs(23,id);
  K_c[70] = exp(-dG[70])/prefRuT;

  k_f[71] = exp(log(6080000000.000001)+0.27*logT-(140.9014693796544/T));
   dG[71] =  - gbs(1,id) - gbs(23,id) + gbs(24,id);
  K_c[71] = exp(-dG[71])/prefRuT;

  k_f[72] = 30000000000.000004;
   dG[72] =   gbs(0,id) - gbs(1,id) + gbs(22,id) - gbs(23,id);
  K_c[72] = exp(-dG[72]);

  k_f[73] = exp(log(540000000.0000001)+0.454*logT-(915.8595509677536/T));
   dG[73] =  - gbs(1,id) - gbs(24,id) + gbs(25,id);
  K_c[73] = exp(-dG[73])/prefRuT;

  k_f[74] = exp(log(1325.0000000000002)+2.53*logT-(6159.407090024893/T));
   dG[74] =   gbs(0,id) - gbs(1,id) + gbs(23,id) - gbs(24,id);
  K_c[74] = exp(-dG[74]);

  k_f[75] = exp(log(521000000000000.06)-0.99*logT-(795.0868629280499/T));
   dG[75] =  - gbs(1,id) - gbs(25,id) + gbs(26,id);
  K_c[75] = exp(-dG[75])/prefRuT;

  k_f[76] = 2000000000.0000002;
   dG[76] =   gbs(0,id) - gbs(1,id) + gbs(24,id) - gbs(25,id);
  K_c[76] = exp(-dG[76]);

  k_f[77] = exp(log(115000.00000000001)+1.9*logT-(3789.243087245706/T));
   dG[77] =   gbs(0,id) - gbs(1,id) + gbs(25,id) - gbs(26,id);
  K_c[77] = exp(-dG[77]);

  k_f[78] = 100000000000.00002;
   dG[78] =  - gbs(1,id) + gbs(11,id) + gbs(14,id) - gbs(27,id);
  K_c[78] = exp(-dG[78]);

  k_f[79] = exp(log(50000000000.00001)-(4025.7562679901257/T));
   dG[79] =   gbs(0,id) - gbs(1,id) + gbs(27,id) - gbs(28,id);
  K_c[79] = exp(-dG[79]);

  k_f[80] = exp(log(11300000000.000002)-(1725.036560833769/T));
   dG[80] =  - gbs(1,id) + gbs(12,id) + gbs(14,id) - gbs(28,id);
  K_c[80] = exp(-dG[80]);

  k_f[81] = 10000000000.000002;
   dG[81] =   gbs(28,id) - gbs(29,id);
  K_c[81] = exp(-dG[81]);

  k_f[82] = exp(log(43000.00000000001)+1.5*logT-(40056.27486650175/T));
   dG[82] =  - gbs(0,id) - gbs(14,id) + gbs(17,id);
  K_c[82] = exp(-dG[82])/prefRuT;

  k_f[83] = exp(log(216000.00000000003)+1.51*logT-(1726.0429999007665/T));
   dG[83] =  - gbs(0,id) + gbs(1,id) - gbs(4,id) + gbs(5,id);
  K_c[83] = exp(-dG[83]);

  k_f[84] = exp(log(74000000000.00002)-0.37*logT);
   dG[84] =  -2.0*gbs(4,id) + gbs(7,id);
  K_c[84] = exp(-dG[84])/prefRuT;

  k_f[85] = exp(log(35.7)+2.4*logT-(-1061.7932156823956/T));
   dG[85] =   gbs(2,id) -2.0*gbs(4,id) + gbs(5,id);
  K_c[85] = exp(-dG[85]);

  k_f[86] = exp(log(14500000000.000002)-(-251.60976674938286/T));
   dG[86] =   gbs(3,id) - gbs(4,id) + gbs(5,id) - gbs(6,id);
  K_c[86] = exp(-dG[86]);

  k_f[87] = exp(log(2000000000.0000002)-(214.87474080397297/T));
   dG[87] =  - gbs(4,id) + gbs(5,id) + gbs(6,id) - gbs(7,id);
  K_c[87] = exp(-dG[87]);

  k_f[88] = exp(log(1700000000000000.2)-(14799.686480198701/T));
   dG[88] =  - gbs(4,id) + gbs(5,id) + gbs(6,id) - gbs(7,id);
  K_c[88] = exp(-dG[88]);

  k_f[89] = 50000000000.00001;
   dG[89] =   gbs(1,id) - gbs(4,id) - gbs(8,id) + gbs(14,id);
  K_c[89] = exp(-dG[89]);

  k_f[90] = 30000000000.000004;
   dG[90] =   gbs(1,id) - gbs(4,id) - gbs(9,id) + gbs(16,id);
  K_c[90] = exp(-dG[90]);

  k_f[91] = 20000000000.000004;
   dG[91] =   gbs(1,id) - gbs(4,id) - gbs(10,id) + gbs(17,id);
  K_c[91] = exp(-dG[91]);

  k_f[92] = exp(log(11300.000000000002)+2*logT-(1509.6586004962971/T));
   dG[92] =  - gbs(4,id) + gbs(5,id) + gbs(9,id) - gbs(10,id);
  K_c[92] = exp(-dG[92]);

  k_f[93] = 30000000000.000004;
   dG[93] =   gbs(1,id) - gbs(4,id) - gbs(11,id) + gbs(17,id);
  K_c[93] = exp(-dG[93]);

  k_f[94] = exp(log(2790000000000000.5)-1.43*logT-(669.2819795533584/T));
   dG[94] =  - gbs(4,id) - gbs(12,id) + gbs(20,id);
  K_c[94] = exp(-dG[94])/prefRuT;

  k_f[95] = exp(log(56000.00000000001)+1.6*logT-(2727.4498715633104/T));
   dG[95] =  - gbs(4,id) + gbs(5,id) + gbs(10,id) - gbs(12,id);
  K_c[95] = exp(-dG[95]);

  k_f[96] = exp(log(644000000000000.1)-1.34*logT-(713.0620789677511/T));
   dG[96] =  - gbs(4,id) + gbs(5,id) + gbs(11,id) - gbs(12,id);
  K_c[96] = exp(-dG[96]);

  k_f[97] = exp(log(100000.00000000001)+1.6*logT-(1570.0449445161491/T));
   dG[97] =  - gbs(4,id) + gbs(5,id) + gbs(12,id) - gbs(13,id);
  K_c[97] = exp(-dG[97]);

  k_f[98] = exp(log(47600.00000000001)+1.228*logT-(35.2253673449136/T));
   dG[98] =   gbs(1,id) - gbs(4,id) - gbs(14,id) + gbs(15,id);
  K_c[98] = exp(-dG[98]);

  k_f[99] = 50000000000.00001;
   dG[99] =  - gbs(4,id) + gbs(5,id) + gbs(14,id) - gbs(16,id);
  K_c[99] = exp(-dG[99]);

  k_f[100] = exp(log(3430000.0000000005)+1.18*logT-(-224.9391314739483/T));
   dG[100] =  - gbs(4,id) + gbs(5,id) + gbs(16,id) - gbs(17,id);
  K_c[100] = exp(-dG[100]);

  k_f[101] = 5000000000.000001;
   dG[101] =  - gbs(4,id) + gbs(5,id) + gbs(17,id) - gbs(18,id);
  K_c[101] = exp(-dG[101]);

  k_f[102] = 5000000000.000001;
   dG[102] =  - gbs(4,id) + gbs(5,id) + gbs(17,id) - gbs(19,id);
  K_c[102] = exp(-dG[102]);

  k_f[103] = exp(log(1440.0000000000002)+2*logT-(-422.70440813896323/T));
   dG[103] =  - gbs(4,id) + gbs(5,id) + gbs(18,id) - gbs(20,id);
  K_c[103] = exp(-dG[103]);

  k_f[104] = exp(log(6300.000000000001)+2*logT-(754.8293002481486/T));
   dG[104] =  - gbs(4,id) + gbs(5,id) + gbs(19,id) - gbs(20,id);
  K_c[104] = exp(-dG[104]);

  k_f[105] = 20000000000.000004;
   dG[105] =   gbs(1,id) - gbs(4,id) - gbs(21,id) + gbs(27,id);
  K_c[105] = exp(-dG[105]);

  k_f[106] = exp(log(2.1800000000000005e-07)+4.5*logT-(-503.2195334987657/T));
   dG[106] =   gbs(1,id) - gbs(4,id) - gbs(22,id) + gbs(28,id);
  K_c[106] = exp(-dG[106]);

  k_f[107] = exp(log(504.0000000000001)+2.3*logT-(6793.463702233337/T));
   dG[107] =   gbs(1,id) - gbs(4,id) - gbs(22,id) + gbs(29,id);
  K_c[107] = exp(-dG[107]);

  k_f[108] = exp(log(33700.0)+2*logT-(7045.07346898272/T));
   dG[108] =  - gbs(4,id) + gbs(5,id) + gbs(21,id) - gbs(22,id);
  K_c[108] = exp(-dG[108]);

  k_f[109] = exp(log(4.830000000000001e-07)+4*logT-(-1006.4390669975314/T));
   dG[109] =  - gbs(4,id) + gbs(12,id) + gbs(14,id) - gbs(22,id);
  K_c[109] = exp(-dG[109]);

  k_f[110] = 5000000000.000001;
   dG[110] =  - gbs(4,id) + gbs(5,id) + gbs(22,id) - gbs(23,id);
  K_c[110] = exp(-dG[110]);

  k_f[111] = exp(log(3600.0000000000005)+2*logT-(1258.0488337469144/T));
   dG[111] =  - gbs(4,id) + gbs(5,id) + gbs(23,id) - gbs(24,id);
  K_c[111] = exp(-dG[111]);

  k_f[112] = exp(log(3540.0000000000005)+2.12*logT-(437.8009941439262/T));
   dG[112] =  - gbs(4,id) + gbs(5,id) + gbs(25,id) - gbs(26,id);
  K_c[112] = exp(-dG[112]);

  k_f[113] = exp(log(7500000000.000001)-(1006.4390669975314/T));
   dG[113] =  - gbs(4,id) + gbs(5,id) + gbs(27,id) - gbs(28,id);
  K_c[113] = exp(-dG[113]);

  k_f[114] = exp(log(130000000.00000001)-(-820.2478396029882/T));
   dG[114] =   gbs(3,id) -2.0*gbs(6,id) + gbs(7,id);
  K_c[114] = exp(-dG[114]);

  k_f[115] = exp(log(420000000000.00006)-(6038.634401985189/T));
   dG[115] =   gbs(3,id) -2.0*gbs(6,id) + gbs(7,id);
  K_c[115] = exp(-dG[115]);

  k_f[116] = 20000000000.000004;
   dG[116] =   gbs(4,id) - gbs(6,id) - gbs(10,id) + gbs(17,id);
  K_c[116] = exp(-dG[116]);

  k_f[117] = 1000000000.0000001;
   dG[117] =   gbs(3,id) - gbs(6,id) - gbs(12,id) + gbs(13,id);
  K_c[117] = exp(-dG[117]);

  k_f[118] = 37800000000.00001;
   dG[118] =   gbs(4,id) - gbs(6,id) - gbs(12,id) + gbs(19,id);
  K_c[118] = exp(-dG[118]);

  k_f[119] = exp(log(150000000000.00003)-(11875.980990570872/T));
   dG[119] =   gbs(4,id) - gbs(6,id) - gbs(14,id) + gbs(15,id);
  K_c[119] = exp(-dG[119]);

  k_f[120] = exp(log(5600.000000000001)+2*logT-(6038.634401985189/T));
   dG[120] =  - gbs(6,id) + gbs(7,id) + gbs(16,id) - gbs(17,id);
  K_c[120] = exp(-dG[120]);

  k_f[121] = exp(log(58000000000.00001)-(289.85445129528904/T));
   dG[121] =   gbs(2,id) - gbs(3,id) - gbs(8,id) + gbs(14,id);
  K_c[121] = exp(-dG[121]);

  k_f[122] = 50000000000.00001;
   dG[122] =   gbs(1,id) - gbs(8,id) - gbs(10,id) + gbs(21,id);
  K_c[122] = exp(-dG[122]);

  k_f[123] = 50000000000.00001;
   dG[123] =   gbs(1,id) - gbs(8,id) - gbs(12,id) + gbs(22,id);
  K_c[123] = exp(-dG[123]);

  k_f[124] = 67100000000.00001;
   dG[124] =   gbs(2,id) - gbs(3,id) - gbs(9,id) + gbs(16,id);
  K_c[124] = exp(-dG[124]);

  k_f[125] = exp(log(108000000000.00002)-(1565.0127491811616/T));
   dG[125] =  - gbs(0,id) + gbs(1,id) - gbs(9,id) + gbs(10,id);
  K_c[125] = exp(-dG[125]);

  k_f[126] = exp(log(5710000000.000001)-(-379.93074779156814/T));
   dG[126] =   gbs(1,id) - gbs(5,id) - gbs(9,id) + gbs(17,id);
  K_c[126] = exp(-dG[126]);

  k_f[127] = 40000000000.00001;
   dG[127] =   gbs(1,id) - gbs(9,id) - gbs(10,id) + gbs(22,id);
  K_c[127] = exp(-dG[127]);

  k_f[128] = 30000000000.000004;
   dG[128] =   gbs(1,id) - gbs(9,id) - gbs(12,id) + gbs(23,id);
  K_c[128] = exp(-dG[128]);

  k_f[129] = 60000000000.00001;
   dG[129] =   gbs(1,id) - gbs(9,id) - gbs(13,id) + gbs(24,id);
  K_c[129] = exp(-dG[129]);

  k_f[130] = 50000000000.00001;
   dG[130] =  - gbs(9,id) - gbs(14,id) + gbs(27,id);
  K_c[130] = exp(-dG[130])/prefRuT;

  k_f[131] = exp(log(190000000000.00003)-(7946.842873012509/T));
   dG[131] =  - gbs(9,id) + gbs(14,id) - gbs(15,id) + gbs(16,id);
  K_c[131] = exp(-dG[131]);

  k_f[132] = exp(log(94600000000.00002)-(-259.15805975186436/T));
   dG[132] =   gbs(1,id) - gbs(9,id) - gbs(17,id) + gbs(28,id);
  K_c[132] = exp(-dG[132]);

  k_f[133] = 50000000000.00001;
   dG[133] =  - gbs(9,id) + gbs(14,id) + gbs(22,id) - gbs(27,id);
  K_c[133] = exp(-dG[133]);

  k_f[134] = exp(log(5000000000.000001)-(754.8293002481486/T));
   dG[134] =   gbs(1,id) - gbs(3,id) + gbs(4,id) - gbs(10,id) + gbs(14,id);
  K_c[134] = prefRuT*exp(-dG[134]);

  k_f[135] = exp(log(500.0000000000001)+2*logT-(3638.277227196076/T));
   dG[135] =  - gbs(0,id) + gbs(1,id) - gbs(10,id) + gbs(12,id);
  K_c[135] = exp(-dG[135]);

  k_f[136] = exp(log(1600000000000.0002)-(6010.454108109258/T));
   dG[136] =   gbs(0,id) -2.0*gbs(10,id) + gbs(22,id);
  K_c[136] = exp(-dG[136]);

  k_f[137] = 40000000000.00001;
   dG[137] =   gbs(1,id) - gbs(10,id) - gbs(12,id) + gbs(24,id);
  K_c[137] = exp(-dG[137]);

  k_f[138] = exp(log(2460.0000000000005)+2*logT-(4161.6255420347925/T));
   dG[138] =  - gbs(10,id) +2.0*gbs(12,id) - gbs(13,id);
  K_c[138] = exp(-dG[138]);

  k_f[139] = exp(log(810000000.0000001)+0.5*logT-(2269.5200960794336/T));
   dG[139] =  - gbs(10,id) - gbs(14,id) + gbs(28,id);
  K_c[139] = exp(-dG[139])/prefRuT;

  k_f[140] = 30000000000.000004;
   dG[140] =  - gbs(10,id) + gbs(14,id) + gbs(23,id) - gbs(27,id);
  K_c[140] = exp(-dG[140]);

  k_f[141] = exp(log(15000000000.000002)-(301.93172009925945/T));
   dG[141] =   gbs(10,id) - gbs(11,id);
  K_c[141] = exp(-dG[141]);

  k_f[142] = exp(log(9000000000.000002)-(301.93172009925945/T));
   dG[142] =   gbs(10,id) - gbs(11,id);
  K_c[142] = exp(-dG[142]);

  k_f[143] = 28000000000.000004;
   dG[143] =   gbs(1,id) - gbs(3,id) + gbs(4,id) - gbs(11,id) + gbs(14,id);
  K_c[143] = prefRuT*exp(-dG[143]);

  k_f[144] = 12000000000.000002;
   dG[144] =  - gbs(3,id) + gbs(5,id) - gbs(11,id) + gbs(14,id);
  K_c[144] = exp(-dG[144]);

  k_f[145] = 70000000000.00002;
   dG[145] =  - gbs(0,id) + gbs(1,id) - gbs(11,id) + gbs(12,id);
  K_c[145] = exp(-dG[145]);

  k_f[146] = exp(log(482000000000000.06)-1.16*logT-(576.1863658560868/T));
   dG[146] =  - gbs(5,id) - gbs(11,id) + gbs(20,id);
  K_c[146] = exp(-dG[146])/prefRuT;

  k_f[147] = 30000000000.000004;
   dG[147] =   gbs(10,id) - gbs(11,id);
  K_c[147] = exp(-dG[147]);

  k_f[148] = exp(log(12000000000.000002)-(-286.83513409429645/T));
   dG[148] =   gbs(1,id) - gbs(11,id) - gbs(12,id) + gbs(24,id);
  K_c[148] = exp(-dG[148]);

  k_f[149] = exp(log(16000000000.000002)-(-286.83513409429645/T));
   dG[149] =  - gbs(11,id) +2.0*gbs(12,id) - gbs(13,id);
  K_c[149] = exp(-dG[149]);

  k_f[150] = 9000000000.000002;
   dG[150] =   gbs(10,id) - gbs(11,id);
  K_c[150] = exp(-dG[150]);

  k_f[151] = 7000000000.000001;
   dG[151] =   gbs(10,id) - gbs(11,id);
  K_c[151] = exp(-dG[151]);

  k_f[152] = 14000000000.000002;
   dG[152] =  - gbs(11,id) + gbs(14,id) - gbs(15,id) + gbs(17,id);
  K_c[152] = exp(-dG[152]);

  k_f[153] = exp(log(40000000000.00001)-(-276.77074342432115/T));
   dG[153] =  - gbs(11,id) + gbs(12,id) + gbs(25,id) - gbs(26,id);
  K_c[153] = exp(-dG[153]);

  k_f[154] = exp(log(35600000000.00001)-(15338.13138104238/T));
   dG[154] =   gbs(2,id) - gbs(3,id) - gbs(12,id) + gbs(19,id);
  K_c[154] = exp(-dG[154]);

  k_f[155] = exp(log(2310000000.0000005)-(10222.904823027426/T));
   dG[155] =  - gbs(3,id) + gbs(4,id) - gbs(12,id) + gbs(17,id);
  K_c[155] = exp(-dG[155]);

  k_f[156] = exp(log(24.500000000000004)+2.47*logT-(2606.6771835236063/T));
   dG[156] =   gbs(6,id) - gbs(7,id) - gbs(12,id) + gbs(13,id);
  K_c[156] = exp(-dG[156]);

  k_f[157] = exp(log(67700000000000.01)-1.18*logT-(329.1055749081928/T));
   dG[157] =  -2.0*gbs(12,id) + gbs(26,id);
  K_c[157] = exp(-dG[157])/prefRuT;

  k_f[158] = exp(log(6840000000.000001)+0.1*logT-(5334.127055086917/T));
   dG[158] =   gbs(1,id) -2.0*gbs(12,id) + gbs(25,id);
  K_c[158] = exp(-dG[158]);

  k_f[159] = 26480000000.000004;
   dG[159] =  - gbs(12,id) + gbs(13,id) + gbs(14,id) - gbs(16,id);
  K_c[159] = exp(-dG[159]);

  k_f[160] = exp(log(3.3200000000000003)+2.81*logT-(2948.866466302767/T));
   dG[160] =  - gbs(12,id) + gbs(13,id) + gbs(16,id) - gbs(17,id);
  K_c[160] = exp(-dG[160]);

  k_f[161] = exp(log(30000.000000000004)+1.5*logT-(5002.002162977731/T));
   dG[161] =  - gbs(12,id) + gbs(13,id) + gbs(18,id) - gbs(20,id);
  K_c[161] = exp(-dG[161]);

  k_f[162] = exp(log(10000.000000000002)+1.5*logT-(5002.002162977731/T));
   dG[162] =  - gbs(12,id) + gbs(13,id) + gbs(19,id) - gbs(20,id);
  K_c[162] = exp(-dG[162]);

  k_f[163] = exp(log(227.00000000000003)+2*logT-(4629.619708188645/T));
   dG[163] =  - gbs(12,id) + gbs(13,id) + gbs(23,id) - gbs(24,id);
  K_c[163] = exp(-dG[163]);

  k_f[164] = exp(log(6140.000000000002)+1.74*logT-(5258.644125062102/T));
   dG[164] =  - gbs(12,id) + gbs(13,id) + gbs(25,id) - gbs(26,id);
  K_c[164] = exp(-dG[164]);

  k_f[165] = exp(log(1500000000000000.2)-1.0*logT-(8554.732069479018/T));
   dG[165] =   gbs(1,id) + gbs(14,id) - gbs(16,id);
  K_c[165] = prefRuT*exp(-dG[165]);

  k_f[166] = exp(log(187000000000000.03)-1.0*logT-(8554.732069479018/T));
   dG[166] =   gbs(1,id) + gbs(14,id) - gbs(16,id);
  K_c[166] = prefRuT*exp(-dG[166]);

  k_f[167] = exp(log(13450000000.000002)-(201.2878133995063/T));
   dG[167] =  - gbs(3,id) + gbs(6,id) + gbs(14,id) - gbs(16,id);
  K_c[167] = exp(-dG[167]);

  k_f[168] = exp(log(18000000000.000004)-(452.8975801488892/T));
   dG[168] =  - gbs(3,id) + gbs(6,id) + gbs(17,id) - gbs(18,id);
  K_c[168] = exp(-dG[168]);

  k_f[169] = exp(log(4.2800000000000005e-16)+7.6*logT-(-1776.364953250643/T));
   dG[169] =  - gbs(3,id) + gbs(6,id) + gbs(17,id) - gbs(19,id);
  K_c[169] = exp(-dG[169]);

  k_f[170] = exp(log(10000000000.000002)-(-379.93074779156814/T));
   dG[170] =  - gbs(3,id) + gbs(14,id) + gbs(16,id) - gbs(21,id);
  K_c[170] = exp(-dG[170]);

  k_f[171] = exp(log(56800000.00000001)+0.9*logT-(1002.9165302630402/T));
   dG[171] =  - gbs(0,id) + gbs(1,id) - gbs(21,id) + gbs(22,id);
  K_c[171] = exp(-dG[171]);

  k_f[172] = exp(log(45800000000000.01)-1.39*logT-(510.7678265012472/T));
   dG[172] =  - gbs(3,id) + gbs(16,id) + gbs(17,id) - gbs(23,id);
  K_c[172] = exp(-dG[172]);

  k_f[173] = exp(log(8000000000000.0)+0.44*logT-(43664.358921687905/T));
   dG[173] =   gbs(0,id) + gbs(22,id) - gbs(24,id);
  K_c[173] = prefRuT*exp(-dG[173]);

  k_f[174] = exp(log(840000000.0000001)-(1949.9756923077173/T));
   dG[174] =  - gbs(3,id) + gbs(6,id) + gbs(24,id) - gbs(25,id);
  K_c[174] = exp(-dG[174]);

  k_f[175] = exp(log(3200000000.0000005)-(429.74948160794594/T));
   dG[175] =  - gbs(3,id) + gbs(4,id) +2.0*gbs(14,id) - gbs(27,id);
  K_c[175] = prefRuT*exp(-dG[175]);

  k_f[176] = 10000000000.000002;
   dG[176] =  2.0*gbs(14,id) + gbs(22,id) -2.0*gbs(27,id);
  K_c[176] = prefRuT*exp(-dG[176]);

  k_f[177] = exp(log(27000000000.000004)-(178.64293439206185/T));
   dG[177] =   gbs(2,id) - gbs(30,id) - gbs(35,id) + gbs(47,id);
  K_c[177] = exp(-dG[177]);

  k_f[178] = exp(log(9000000.000000002)+1*logT-(3270.9269677419775/T));
   dG[178] =   gbs(2,id) - gbs(3,id) - gbs(30,id) + gbs(35,id);
  K_c[178] = exp(-dG[178]);

  k_f[179] = exp(log(33600000000.000008)-(193.73952039702482/T));
   dG[179] =   gbs(1,id) - gbs(4,id) - gbs(30,id) + gbs(35,id);
  K_c[179] = exp(-dG[179]);

  k_f[180] = exp(log(1400000000.0000002)-(5439.803157121658/T));
   dG[180] =  - gbs(2,id) + gbs(3,id) - gbs(37,id) + gbs(47,id);
  K_c[180] = exp(-dG[180]);

  k_f[181] = exp(log(29000000000.000004)-(11649.532200496427/T));
   dG[181] =  - gbs(2,id) +2.0*gbs(35,id) - gbs(37,id);
  K_c[181] = exp(-dG[181]);

  k_f[182] = exp(log(387000000000.00006)-(9500.784792456698/T));
   dG[182] =  - gbs(1,id) + gbs(4,id) - gbs(37,id) + gbs(47,id);
  K_c[182] = exp(-dG[182]);

  k_f[183] = exp(log(2000000000.0000002)-(10597.803375484007/T));
   dG[183] =  - gbs(4,id) + gbs(6,id) - gbs(37,id) + gbs(47,id);
  K_c[183] = exp(-dG[183]);

  k_f[184] = exp(log(79100000000.0)-(28190.358266600855/T));
   dG[184] =   gbs(2,id) - gbs(37,id) + gbs(47,id);
  K_c[184] = prefRuT*exp(-dG[184]);

  k_f[185] = exp(log(2110000000.0000005)-(-241.54537607940756/T));
   dG[185] =   gbs(4,id) - gbs(6,id) - gbs(35,id) + gbs(36,id);
  K_c[185] = exp(-dG[185]);

  k_f[186] = exp(log(106000000000000.02)-1.41*logT);
   dG[186] =  - gbs(2,id) - gbs(35,id) + gbs(36,id);
  K_c[186] = exp(-dG[186])/prefRuT;

  k_f[187] = exp(log(3900000000.0000005)-(-120.77268803970378/T));
   dG[187] =  - gbs(2,id) + gbs(3,id) + gbs(35,id) - gbs(36,id);
  K_c[187] = exp(-dG[187]);

  k_f[188] = exp(log(132000000000.00002)-(181.15903205955567/T));
   dG[188] =  - gbs(1,id) + gbs(4,id) + gbs(35,id) - gbs(36,id);
  K_c[188] = exp(-dG[188]);

  k_f[189] = 40000000000.00001;
   dG[189] =   gbs(1,id) - gbs(2,id) - gbs(31,id) + gbs(35,id);
  K_c[189] = exp(-dG[189]);

  k_f[190] = exp(log(32000000000.000004)-(166.0624460545927/T));
   dG[190] =   gbs(0,id) - gbs(1,id) + gbs(30,id) - gbs(31,id);
  K_c[190] = exp(-dG[190]);

  k_f[191] = 20000000000.000004;
   dG[191] =   gbs(1,id) - gbs(4,id) - gbs(31,id) + gbs(38,id);
  K_c[191] = exp(-dG[191]);

  k_f[192] = exp(log(2000000.0000000002)+1.2*logT);
   dG[192] =  - gbs(4,id) + gbs(5,id) + gbs(30,id) - gbs(31,id);
  K_c[192] = exp(-dG[192]);

  k_f[193] = exp(log(461.00000000000006)+2*logT-(3270.9269677419775/T));
   dG[193] =   gbs(2,id) - gbs(3,id) - gbs(31,id) + gbs(38,id);
  K_c[193] = exp(-dG[193]);

  k_f[194] = exp(log(1280.0000000000002)+1.5*logT-(50.32195334987657/T));
   dG[194] =  - gbs(3,id) + gbs(4,id) - gbs(31,id) + gbs(35,id);
  K_c[194] = exp(-dG[194]);

  k_f[195] = 15000000000.000002;
   dG[195] =   gbs(1,id) - gbs(30,id) - gbs(31,id) + gbs(47,id);
  K_c[195] = exp(-dG[195]);

  k_f[196] = exp(log(20000000000.000004)-(6969.590538957906/T));
   dG[196] =   gbs(0,id) - gbs(5,id) - gbs(31,id) + gbs(38,id);
  K_c[196] = exp(-dG[196]);

  k_f[197] = exp(log(21600000000.000004)-0.23*logT);
   dG[197] =   gbs(4,id) - gbs(31,id) - gbs(35,id) + gbs(47,id);
  K_c[197] = exp(-dG[197]);

  k_f[198] = exp(log(365000000000.00006)-0.45*logT);
   dG[198] =   gbs(1,id) - gbs(31,id) - gbs(35,id) + gbs(37,id);
  K_c[198] = exp(-dG[198]);

  k_f[199] = 3000000000.0000005;
   dG[199] =  - gbs(2,id) + gbs(4,id) + gbs(31,id) - gbs(32,id);
  K_c[199] = exp(-dG[199]);

  k_f[200] = 39000000000.00001;
   dG[200] =   gbs(1,id) - gbs(2,id) - gbs(32,id) + gbs(38,id);
  K_c[200] = exp(-dG[200]);

  k_f[201] = exp(log(40000000000.00001)-(1836.751297270495/T));
   dG[201] =   gbs(0,id) - gbs(1,id) + gbs(31,id) - gbs(32,id);
  K_c[201] = exp(-dG[201]);

  k_f[202] = exp(log(90000.00000000001)+1.5*logT-(-231.48098540943224/T));
   dG[202] =  - gbs(4,id) + gbs(5,id) + gbs(31,id) - gbs(32,id);
  K_c[202] = exp(-dG[202]);

  k_f[203] = 330000000.0;
   dG[203] =   gbs(1,id) - gbs(34,id) + gbs(47,id);
  K_c[203] = prefRuT*exp(-dG[203]);

  k_f[204] = exp(log(130000000000.00002)-0.11*logT-(2506.033276823853/T));
   dG[204] =   gbs(1,id) - gbs(34,id) + gbs(47,id);
  K_c[204] = prefRuT*exp(-dG[204]);

  k_f[205] = 5000000000.000001;
   dG[205] =  - gbs(3,id) + gbs(6,id) - gbs(34,id) + gbs(47,id);
  K_c[205] = exp(-dG[205]);

  k_f[206] = 25000000000.000004;
   dG[206] =  - gbs(2,id) + gbs(4,id) - gbs(34,id) + gbs(47,id);
  K_c[206] = exp(-dG[206]);

  k_f[207] = 70000000000.00002;
   dG[207] =  - gbs(2,id) + gbs(31,id) - gbs(34,id) + gbs(35,id);
  K_c[207] = exp(-dG[207]);

  k_f[208] = 50000000000.00001;
   dG[208] =   gbs(0,id) - gbs(1,id) - gbs(34,id) + gbs(47,id);
  K_c[208] = exp(-dG[208]);

  k_f[209] = 20000000000.000004;
   dG[209] =  - gbs(4,id) + gbs(5,id) - gbs(34,id) + gbs(47,id);
  K_c[209] = exp(-dG[209]);

  k_f[210] = 25000000000.000004;
   dG[210] =  - gbs(12,id) + gbs(13,id) - gbs(34,id) + gbs(47,id);
  K_c[210] = exp(-dG[210]);

  k_f[211] = exp(log(44800000000000.01)-1.32*logT-(372.38245478908664/T));
   dG[211] =  - gbs(1,id) - gbs(35,id) + gbs(38,id);
  K_c[211] = exp(-dG[211])/prefRuT;

  k_f[212] = 25000000000.000004;
   dG[212] =  - gbs(2,id) + gbs(4,id) + gbs(35,id) - gbs(38,id);
  K_c[212] = exp(-dG[212]);

  k_f[213] = exp(log(900000000.0000001)+0.72*logT-(332.1248921091854/T));
   dG[213] =   gbs(0,id) - gbs(1,id) + gbs(35,id) - gbs(38,id);
  K_c[213] = exp(-dG[213]);

  k_f[214] = exp(log(13000.000000000002)+1.9*logT-(-478.0585568238275/T));
   dG[214] =  - gbs(4,id) + gbs(5,id) + gbs(35,id) - gbs(38,id);
  K_c[214] = exp(-dG[214]);

  k_f[215] = exp(log(10000000000.000002)-(6541.853935483955/T));
   dG[215] =  - gbs(3,id) + gbs(6,id) + gbs(35,id) - gbs(38,id);
  K_c[215] = exp(-dG[215]);

  k_f[216] = 77000000000.00002;
   dG[216] =  - gbs(2,id) + gbs(14,id) + gbs(30,id) - gbs(39,id);
  K_c[216] = exp(-dG[216]);

  k_f[217] = 40000000000.00001;
   dG[217] =   gbs(1,id) - gbs(4,id) - gbs(39,id) + gbs(46,id);
  K_c[217] = exp(-dG[217]);

  k_f[218] = exp(log(8000000000.000001)-(3754.0177199007926/T));
   dG[218] =   gbs(4,id) - gbs(5,id) - gbs(39,id) + gbs(40,id);
  K_c[218] = exp(-dG[218]);

  k_f[219] = exp(log(6140000000.000001)-(-221.4165947394569/T));
   dG[219] =   gbs(2,id) - gbs(3,id) - gbs(39,id) + gbs(46,id);
  K_c[219] = exp(-dG[219]);

  k_f[220] = exp(log(295.00000000000006)+2.45*logT-(1127.2117550372352/T));
   dG[220] =  - gbs(0,id) + gbs(1,id) - gbs(39,id) + gbs(40,id);
  K_c[220] = exp(-dG[220]);

  k_f[221] = 23500000000.000004;
   dG[221] =  - gbs(2,id) + gbs(14,id) + gbs(35,id) - gbs(46,id);
  K_c[221] = exp(-dG[221]);

  k_f[222] = 54000000000.00001;
   dG[222] =  - gbs(1,id) + gbs(14,id) + gbs(31,id) - gbs(46,id);
  K_c[222] = exp(-dG[222]);

  k_f[223] = 2500000000.0000005;
   dG[223] =   gbs(1,id) - gbs(4,id) + gbs(14,id) + gbs(35,id) - gbs(46,id);
  K_c[223] = prefRuT*exp(-dG[223]);

  k_f[224] = 20000000000.000004;
   dG[224] =   gbs(14,id) - gbs(30,id) - gbs(46,id) + gbs(47,id);
  K_c[224] = exp(-dG[224]);

  k_f[225] = exp(log(2000000000.0000002)-(10064.390669975315/T));
   dG[225] =  - gbs(3,id) + gbs(15,id) + gbs(35,id) - gbs(46,id);
  K_c[225] = exp(-dG[225]);

  k_f[226] = exp(log(310000000000.00006)-(27199.015785608288/T));
   dG[226] =   gbs(14,id) + gbs(30,id) - gbs(46,id);
  K_c[226] = prefRuT*exp(-dG[226]);

  k_f[227] = exp(log(190000000000000.03)-1.52*logT-(372.38245478908664/T));
   dG[227] =   gbs(14,id) - gbs(35,id) + gbs(37,id) - gbs(46,id);
  K_c[227] = exp(-dG[227]);

  k_f[228] = exp(log(3800000000000000.5)-2*logT-(402.5756267990126/T));
   dG[228] =   gbs(15,id) - gbs(35,id) - gbs(46,id) + gbs(47,id);
  K_c[228] = exp(-dG[228]);

  k_f[229] = exp(log(1.0400000000000003e+26)-3.3*logT-(63707.592940943745/T));
   dG[229] =   gbs(1,id) + gbs(39,id) - gbs(40,id);
  K_c[229] = prefRuT*exp(-dG[229]);

  k_f[230] = exp(log(20.3)+2.64*logT-(2506.033276823853/T));
   dG[230] =   gbs(1,id) - gbs(2,id) - gbs(40,id) + gbs(46,id);
  K_c[230] = exp(-dG[230]);

  k_f[231] = exp(log(5.07)+2.64*logT-(2506.033276823853/T));
   dG[231] =  - gbs(2,id) + gbs(14,id) + gbs(31,id) - gbs(40,id);
  K_c[231] = exp(-dG[231]);

  k_f[232] = exp(log(3910000.0000000005)+1.58*logT-(13385.639591067169/T));
   dG[232] =  - gbs(2,id) + gbs(4,id) + gbs(39,id) - gbs(40,id);
  K_c[232] = exp(-dG[232]);

  k_f[233] = exp(log(1100.0)+2.03*logT-(6728.045162878498/T));
   dG[233] =   gbs(1,id) - gbs(4,id) - gbs(40,id) + gbs(44,id);
  K_c[233] = exp(-dG[233]);

  k_f[234] = exp(log(4.400000000000001)+2.26*logT-(3220.6050143921007/T));
   dG[234] =   gbs(1,id) - gbs(4,id) - gbs(40,id) + gbs(45,id);
  K_c[234] = exp(-dG[234]);

  k_f[235] = exp(log(0.16000000000000003)+2.56*logT-(4528.975801488891/T));
   dG[235] =  - gbs(4,id) + gbs(14,id) + gbs(32,id) - gbs(40,id);
  K_c[235] = exp(-dG[235]);

  k_f[236] = 33000000000.000004;
   dG[236] =  - gbs(1,id) - gbs(40,id) + gbs(41,id);
  K_c[236] = exp(-dG[236])/prefRuT;

  k_f[237] = exp(log(60000000000.00001)-(201.2878133995063/T));
   dG[237] =   gbs(10,id) - gbs(30,id) - gbs(41,id) + gbs(47,id);
  K_c[237] = exp(-dG[237]);

  k_f[238] = exp(log(63000000000.00001)-(23158.1629316132/T));
   dG[238] =  - gbs(8,id) + gbs(30,id) + gbs(39,id) - gbs(47,id);
  K_c[238] = exp(-dG[238]);

  k_f[239] = exp(log(3120000.0000000005)+0.88*logT-(10129.809209330155/T));
   dG[239] =  - gbs(9,id) + gbs(30,id) + gbs(40,id) - gbs(47,id);
  K_c[239] = exp(-dG[239]);

  k_f[240] = exp(log(3100000000.0000005)+0.15*logT);
   dG[240] =  - gbs(9,id) + gbs(42,id) - gbs(47,id);
  K_c[240] = exp(-dG[240])/prefRuT;

  k_f[241] = exp(log(10000000000.000002)-(37238.24547890866/T));
   dG[241] =  - gbs(10,id) + gbs(31,id) + gbs(40,id) - gbs(47,id);
  K_c[241] = exp(-dG[241]);

  k_f[242] = exp(log(100000000.00000001)-(32709.269677419772/T));
   dG[242] =  - gbs(11,id) + gbs(31,id) + gbs(40,id) - gbs(47,id);
  K_c[242] = exp(-dG[242]);

  k_f[243] = 19000000000.000004;
   dG[243] =   gbs(2,id) - gbs(8,id) - gbs(35,id) + gbs(39,id);
  K_c[243] = exp(-dG[243]);

  k_f[244] = 29000000000.000004;
   dG[244] =  - gbs(8,id) + gbs(14,id) + gbs(30,id) - gbs(35,id);
  K_c[244] = exp(-dG[244]);

  k_f[245] = 41000000000.00001;
   dG[245] =   gbs(2,id) - gbs(9,id) - gbs(35,id) + gbs(40,id);
  K_c[245] = exp(-dG[245]);

  k_f[246] = 16200000000.000002;
   dG[246] =   gbs(1,id) - gbs(9,id) - gbs(35,id) + gbs(46,id);
  K_c[246] = exp(-dG[246]);

  k_f[247] = 24600000000.000004;
   dG[247] =  - gbs(9,id) + gbs(16,id) + gbs(30,id) - gbs(35,id);
  K_c[247] = exp(-dG[247]);

  k_f[248] = exp(log(310000000000000.06)-1.38*logT-(639.0888075434325/T));
   dG[248] =   gbs(1,id) - gbs(10,id) - gbs(35,id) + gbs(45,id);
  K_c[248] = exp(-dG[248]);

  k_f[249] = exp(log(290000000000.00006)-0.69*logT-(382.44684545906193/T));
   dG[249] =   gbs(4,id) - gbs(10,id) - gbs(35,id) + gbs(40,id);
  K_c[249] = exp(-dG[249]);

  k_f[250] = exp(log(38000000000.00001)-0.36*logT-(291.86732942928415/T));
   dG[250] =   gbs(1,id) - gbs(10,id) - gbs(35,id) + gbs(43,id);
  K_c[250] = exp(-dG[250]);

  k_f[251] = exp(log(310000000000000.06)-1.38*logT-(639.0888075434325/T));
   dG[251] =   gbs(1,id) - gbs(11,id) - gbs(35,id) + gbs(45,id);
  K_c[251] = exp(-dG[251]);

  k_f[252] = exp(log(290000000000.00006)-0.69*logT-(382.44684545906193/T));
   dG[252] =   gbs(4,id) - gbs(11,id) - gbs(35,id) + gbs(40,id);
  K_c[252] = exp(-dG[252]);

  k_f[253] = exp(log(38000000000.00001)-0.36*logT-(291.86732942928415/T));
   dG[253] =   gbs(1,id) - gbs(11,id) - gbs(35,id) + gbs(43,id);
  K_c[253] = exp(-dG[253]);

  k_f[254] = exp(log(96000000000.00002)-(14492.722564764454/T));
   dG[254] =   gbs(5,id) - gbs(12,id) - gbs(35,id) + gbs(40,id);
  K_c[254] = exp(-dG[254]);

  k_f[255] = exp(log(1000000000.0000001)-(10945.024853598155/T));
   dG[255] =   gbs(4,id) - gbs(12,id) - gbs(35,id) + gbs(41,id);
  K_c[255] = exp(-dG[255]);

  k_f[256] = 22000000000.000004;
   dG[256] =   gbs(1,id) - gbs(2,id) + gbs(14,id) - gbs(42,id) + gbs(47,id);
  K_c[256] = prefRuT*exp(-dG[256]);

  k_f[257] = 2000000000.0000002;
   dG[257] =  - gbs(2,id) + gbs(35,id) + gbs(40,id) - gbs(42,id);
  K_c[257] = exp(-dG[257]);

  k_f[258] = 12000000000.000002;
   dG[258] =   gbs(2,id) - gbs(3,id) + gbs(16,id) - gbs(42,id) + gbs(47,id);
  K_c[258] = prefRuT*exp(-dG[258]);

  k_f[259] = 12000000000.000002;
   dG[259] =   gbs(1,id) - gbs(4,id) + gbs(16,id) - gbs(42,id) + gbs(47,id);
  K_c[259] = prefRuT*exp(-dG[259]);

  k_f[260] = 100000000000.00002;
   dG[260] =  - gbs(1,id) + gbs(10,id) - gbs(42,id) + gbs(47,id);
  K_c[260] = exp(-dG[260]);

  k_f[261] = exp(log(98000.00000000001)+1.41*logT-(4277.366034739509/T));
   dG[261] =  - gbs(2,id) + gbs(15,id) + gbs(31,id) - gbs(45,id);
  K_c[261] = exp(-dG[261]);

  k_f[262] = exp(log(150000.00000000003)+1.57*logT-(22141.659473945692/T));
   dG[262] =  - gbs(2,id) + gbs(14,id) + gbs(38,id) - gbs(45,id);
  K_c[262] = exp(-dG[262]);

  k_f[263] = exp(log(2200.0)+2.11*logT-(5736.702681885929/T));
   dG[263] =  - gbs(2,id) + gbs(4,id) - gbs(45,id) + gbs(46,id);
  K_c[263] = exp(-dG[263]);

  k_f[264] = exp(log(22500.000000000004)+1.7*logT-(1912.23422729531/T));
   dG[264] =  - gbs(1,id) + gbs(14,id) + gbs(32,id) - gbs(45,id);
  K_c[264] = exp(-dG[264]);

  k_f[265] = exp(log(105.00000000000003)+2.5*logT-(6692.8197955335845/T));
   dG[265] =   gbs(0,id) - gbs(1,id) - gbs(45,id) + gbs(46,id);
  K_c[265] = exp(-dG[265]);

  k_f[266] = exp(log(33000.00000000001)+1.5*logT-(1811.5903205955567/T));
   dG[266] =  - gbs(4,id) + gbs(5,id) - gbs(45,id) + gbs(46,id);
  K_c[266] = exp(-dG[266]);

  k_f[267] = exp(log(3300.000000000001)+1.5*logT-(1811.5903205955567/T));
   dG[267] =  - gbs(4,id) + gbs(15,id) + gbs(32,id) - gbs(45,id);
  K_c[267] = exp(-dG[267]);

  k_f[268] = exp(log(11800000000000.002)-(42632.75887801543/T));
   dG[268] =   gbs(14,id) + gbs(31,id) - gbs(45,id);
  K_c[268] = prefRuT*exp(-dG[268]);

  k_f[269] = exp(log(2100000000000.0002)-0.69*logT-(1434.1756704714824/T));
   dG[269] =  - gbs(43,id) + gbs(45,id);
  K_c[269] = exp(-dG[269]);

  k_f[270] = exp(log(270000000.00000006)+0.18*logT-(1066.8254110173834/T));
   dG[270] =  - gbs(1,id) + gbs(4,id) + gbs(40,id) - gbs(43,id);
  K_c[270] = exp(-dG[270]);

  k_f[271] = exp(log(170000000000.00003)-0.75*logT-(1454.304451811433/T));
   dG[271] =  - gbs(1,id) + gbs(14,id) + gbs(32,id) - gbs(43,id);
  K_c[271] = exp(-dG[271]);

  k_f[272] = exp(log(20000.000000000004)+2*logT-(1006.4390669975314/T));
   dG[272] =  - gbs(44,id) + gbs(45,id);
  K_c[272] = exp(-dG[272]);

  k_f[273] = 9000000000.000002;
   dG[273] =   gbs(14,id) - gbs(27,id) - gbs(35,id) + gbs(43,id);
  K_c[273] = exp(-dG[273]);

  k_f[274] = exp(log(610000000000.0001)-0.31*logT-(145.93366471464208/T));
   dG[274] =   gbs(1,id) - gbs(12,id) - gbs(30,id) + gbs(41,id);
  K_c[274] = exp(-dG[274]);

  k_f[275] = exp(log(3700000000.0000005)+0.15*logT-(-45.28975801488892/T));
   dG[275] =   gbs(0,id) - gbs(12,id) - gbs(30,id) + gbs(40,id);
  K_c[275] = exp(-dG[275]);

  k_f[276] = exp(log(540.0)+2.4*logT-(4989.421674640263/T));
   dG[276] =   gbs(0,id) - gbs(1,id) + gbs(32,id) - gbs(33,id);
  K_c[276] = exp(-dG[276]);

  k_f[277] = exp(log(50000.00000000001)+1.6*logT-(480.57465449132127/T));
   dG[277] =  - gbs(4,id) + gbs(5,id) + gbs(32,id) - gbs(33,id);
  K_c[277] = exp(-dG[277]);

  k_f[278] = exp(log(9400.000000000002)+1.94*logT-(3250.7981864020267/T));
   dG[278] =  - gbs(2,id) + gbs(4,id) + gbs(32,id) - gbs(33,id);
  K_c[278] = exp(-dG[278]);

  k_f[279] = exp(log(10000000000.000002)-(7221.200305707288/T));
   dG[279] =   gbs(14,id) - gbs(15,id) - gbs(31,id) + gbs(38,id);
  K_c[279] = exp(-dG[279]);

  k_f[280] = exp(log(6160000000000.001)-0.752*logT-(173.61073905707417/T));
   dG[280] =   gbs(35,id) - gbs(36,id) - gbs(39,id) + gbs(46,id);
  K_c[280] = exp(-dG[280]);

  k_f[281] = exp(log(3250000000.0000005)-(-354.76977111662984/T));
   dG[281] =   gbs(15,id) - gbs(36,id) + gbs(37,id) - gbs(46,id);
  K_c[281] = exp(-dG[281]);

  k_f[282] = exp(log(3000000000.0000005)-(5686.380728536053/T));
   dG[282] =   gbs(14,id) - gbs(15,id) - gbs(30,id) + gbs(35,id);
  K_c[282] = exp(-dG[282]);

  k_f[283] = 33700000000.000008;
   dG[283] =   gbs(0,id) + gbs(1,id) - gbs(2,id) - gbs(12,id) + gbs(14,id);
  K_c[283] = prefRuT*exp(-dG[283]);

  k_f[284] = exp(log(6700.000000000001)+1.83*logT-(110.70829736972846/T));
   dG[284] =   gbs(1,id) - gbs(2,id) - gbs(24,id) + gbs(51,id);
  K_c[284] = exp(-dG[284]);

  k_f[285] = 109600000000.00002;
   dG[285] =   gbs(1,id) - gbs(2,id) - gbs(25,id) + gbs(52,id);
  K_c[285] = exp(-dG[285]);

  k_f[286] = exp(log(5000000000000.001)-(8720.794515533611/T));
   dG[286] =   gbs(3,id) - gbs(4,id) + gbs(5,id) - gbs(6,id);
  K_c[286] = exp(-dG[286]);

  k_f[287] = exp(log(8000000.000000001)+0.5*logT-(-883.1502812903339/T));
   dG[287] =   gbs(0,id) - gbs(4,id) - gbs(12,id) + gbs(17,id);
  K_c[287] = exp(-dG[287]);

  k_f[288] = exp(log(1970000000.0000002)+0.43*logT-(-186.19122739454332/T));
   dG[288] =  - gbs(0,id) - gbs(9,id) + gbs(12,id);
  K_c[288] = exp(-dG[288])/prefRuT;

  k_f[289] = exp(log(5800000000.000001)-(754.8293002481486/T));
   dG[289] =  2.0*gbs(1,id) - gbs(3,id) - gbs(10,id) + gbs(15,id);
  K_c[289] = prefRuT*exp(-dG[289]);

  k_f[290] = exp(log(2400000000.0000005)-(754.8293002481486/T));
   dG[290] =   gbs(2,id) - gbs(3,id) - gbs(10,id) + gbs(17,id);
  K_c[290] = exp(-dG[290]);

  k_f[291] = exp(log(200000000000.00003)-(5529.879453617937/T));
   dG[291] =  2.0*gbs(1,id) -2.0*gbs(10,id) + gbs(22,id);
  K_c[291] = prefRuT*exp(-dG[291]);

  k_f[292] = exp(log(68200000.00000001)+0.25*logT-(-470.510263821346/T));
   dG[292] =   gbs(0,id) - gbs(5,id) - gbs(11,id) + gbs(17,id);
  K_c[292] = exp(-dG[292]);

  k_f[293] = exp(log(303000000.00000006)+0.29*logT-(5.5354148684864235/T));
   dG[293] =   gbs(2,id) - gbs(3,id) - gbs(23,id) + gbs(51,id);
  K_c[293] = exp(-dG[293]);

  k_f[294] = exp(log(1337.0000000000002)+1.61*logT-(-193.23630086352605/T));
   dG[294] =  - gbs(3,id) + gbs(6,id) + gbs(22,id) - gbs(23,id);
  K_c[294] = exp(-dG[294]);

  k_f[295] = exp(log(2920000000.0000005)-(909.8209165657685/T));
   dG[295] =  - gbs(2,id) + gbs(4,id) + gbs(51,id) - gbs(52,id);
  K_c[295] = exp(-dG[295]);

  k_f[296] = exp(log(2920000000.0000005)-(909.8209165657685/T));
   dG[296] =  - gbs(2,id) + gbs(4,id) + gbs(12,id) + gbs(14,id) - gbs(52,id);
  K_c[296] = prefRuT*exp(-dG[296]);

  k_f[297] = exp(log(30100000000.000004)-(19701.044736476677/T));
   dG[297] =  - gbs(3,id) + gbs(6,id) + gbs(12,id) + gbs(14,id) - gbs(52,id);
  K_c[297] = prefRuT*exp(-dG[297]);

  k_f[298] = exp(log(2050000.0000000005)+1.16*logT-(1210.2429780645316/T));
   dG[298] =   gbs(0,id) - gbs(1,id) + gbs(51,id) - gbs(52,id);
  K_c[298] = exp(-dG[298]);

  k_f[299] = exp(log(2050000.0000000005)+1.16*logT-(1210.2429780645316/T));
   dG[299] =   gbs(0,id) - gbs(1,id) + gbs(12,id) + gbs(14,id) - gbs(52,id);
  K_c[299] = prefRuT*exp(-dG[299]);

  k_f[300] = exp(log(23430000.000000004)+0.73*logT-(-560.0833407841262/T));
   dG[300] =  - gbs(4,id) + gbs(5,id) + gbs(12,id) + gbs(14,id) - gbs(52,id);
  K_c[300] = prefRuT*exp(-dG[300]);

  k_f[301] = exp(log(3010000000.0000005)-(5999.8864979057835/T));
   dG[301] =  - gbs(6,id) + gbs(7,id) + gbs(12,id) + gbs(14,id) - gbs(52,id);
  K_c[301] = prefRuT*exp(-dG[301]);

  k_f[302] = exp(log(2720.0000000000005)+1.77*logT-(2979.059638312693/T));
   dG[302] =   gbs(13,id) + gbs(14,id) - gbs(52,id);
  K_c[302] = prefRuT*exp(-dG[302]);

  k_f[303] = exp(log(486500000.00000006)+0.422*logT-(-883.1502812903339/T));
   dG[303] =  - gbs(1,id) - gbs(28,id) + gbs(51,id);
  K_c[303] = exp(-dG[303])/prefRuT;

  k_f[304] = 150000000000.00003;
   dG[304] =   gbs(1,id) - gbs(2,id) + gbs(10,id) + gbs(15,id) - gbs(51,id);
  K_c[304] = prefRuT*exp(-dG[304]);

  k_f[305] = 18100000.000000004;
   dG[305] =  - gbs(3,id) + gbs(4,id) + gbs(14,id) + gbs(17,id) - gbs(51,id);
  K_c[305] = prefRuT*exp(-dG[305]);

  k_f[306] = 23500000.000000004;
   dG[306] =  - gbs(3,id) + gbs(4,id) +2.0*gbs(16,id) - gbs(51,id);
  K_c[306] = prefRuT*exp(-dG[306]);

  k_f[307] = 22000000000.000004;
   dG[307] =  - gbs(1,id) + gbs(12,id) + gbs(16,id) - gbs(51,id);
  K_c[307] = exp(-dG[307]);

  k_f[308] = 11000000000.000002;
   dG[308] =   gbs(0,id) - gbs(1,id) + gbs(28,id) - gbs(51,id);
  K_c[308] = exp(-dG[308]);

  k_f[309] = 12000000000.000002;
   dG[309] =  - gbs(4,id) + gbs(5,id) + gbs(28,id) - gbs(51,id);
  K_c[309] = exp(-dG[309]);

  k_f[310] = 30100000000.000004;
   dG[310] =  - gbs(4,id) + gbs(16,id) + gbs(18,id) - gbs(51,id);
  K_c[310] = exp(-dG[310]);

  k_f[311] = 9430000000.000002;
   dG[311] =  - gbs(12,id) - gbs(25,id) + gbs(50,id);
  K_c[311] = exp(-dG[311])/prefRuT;

  k_f[312] = exp(log(193.00000000000003)+2.68*logT-(1869.9637864814135/T));
   dG[312] =  - gbs(2,id) + gbs(4,id) + gbs(49,id) - gbs(50,id);
  K_c[312] = exp(-dG[312]);

  k_f[313] = exp(log(1320.0000000000002)+2.54*logT-(3399.7511683176613/T));
   dG[313] =   gbs(0,id) - gbs(1,id) + gbs(49,id) - gbs(50,id);
  K_c[313] = exp(-dG[313]);

  k_f[314] = exp(log(31600.000000000004)+1.8*logT-(470.0070442878472/T));
   dG[314] =  - gbs(4,id) + gbs(5,id) + gbs(49,id) - gbs(50,id);
  K_c[314] = exp(-dG[314]);

  k_f[315] = exp(log(0.37800000000000006)+2.72*logT-(754.8293002481486/T));
   dG[315] =   gbs(6,id) - gbs(7,id) - gbs(49,id) + gbs(50,id);
  K_c[315] = exp(-dG[315]);

  k_f[316] = exp(log(0.0009030000000000002)+3.65*logT-(3600.03254265017/T));
   dG[316] =  - gbs(12,id) + gbs(13,id) + gbs(49,id) - gbs(50,id);
  K_c[316] = exp(-dG[316]);

  k_f[317] = exp(log(2550.0000000000005)+1.6*logT-(2868.3513409429647/T));
   dG[317] =  - gbs(12,id) - gbs(24,id) + gbs(49,id);
  K_c[317] = exp(-dG[317])/prefRuT;

  k_f[318] = 96400000000.00002;
   dG[318] =  - gbs(2,id) + gbs(17,id) + gbs(25,id) - gbs(49,id);
  K_c[318] = exp(-dG[318]);

  k_f[319] = 36130000000.00001;
   dG[319] =  - gbs(1,id) - gbs(49,id) + gbs(50,id);
  K_c[319] = exp(-dG[319])/prefRuT;

  k_f[320] = exp(log(4060.0000000000005)+2.19*logT-(447.8653848139015/T));
   dG[320] =  - gbs(1,id) + gbs(12,id) + gbs(25,id) - gbs(49,id);
  K_c[320] = exp(-dG[320]);

  k_f[321] = 24100000000.000004;
   dG[321] =  - gbs(4,id) + gbs(18,id) + gbs(25,id) - gbs(49,id);
  K_c[321] = exp(-dG[321]);

  k_f[322] = exp(log(25500000.000000004)+0.255*logT-(-474.5360200893361/T));
   dG[322] =   gbs(3,id) - gbs(6,id) - gbs(49,id) + gbs(50,id);
  K_c[322] = exp(-dG[322]);

  k_f[323] = 24100000000.000004;
   dG[323] =   gbs(4,id) - gbs(6,id) + gbs(17,id) + gbs(25,id) - gbs(49,id);
  K_c[323] = prefRuT*exp(-dG[323]);

  k_f[324] = exp(log(19270000000.000004)-0.32*logT);
   dG[324] =  - gbs(12,id) +2.0*gbs(25,id) - gbs(49,id);
  K_c[324] = exp(-dG[324]);

  // ----------------------------------------------------------- >
  // FallOff Modifications. ------------------------------------ >
  // ----------------------------------------------------------- >

  double Fcent[50];
  double pmod[50];
  double Pr,k0;
  double A,f1,F_pdr;
  double C,N;

  //  Three Body Reaction #0
  //  Three Body Reaction #1
  //  Lindeman Reaction #11
  Fcent[2] = 1.0;
  k0 = exp(log(602000000.0000001)-(1509.6586004962971/T));
  Pr = S_tbc[11]*k0/k_f[11];
  pmod[2] = Pr/(1.0 + Pr);
  k_f[11] = k_f[11]*pmod[2];

  //  Three Body Reaction #32
  //  Three Body Reaction #33
  //  Three Body Reaction #34
  //  Three Body Reaction #35
  //  Three Body Reaction #36
  //  Three Body Reaction #38
  //  Three Body Reaction #39
  //  Three Body Reaction #40
  //  Three Body Reaction #41
  //  Three Body Reaction #42
  //  Troe Reaction #49
  Fcent[13] = (1.0 - (0.562))*exp(-T/(91.0)) + (0.562) *exp(-T/(5836.0)) + exp(-(8552.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[13]);
  N =   0.75 - 1.27*log10(Fcent[13]);
  k0 = exp(log(1.0400000000000002e+20)-2.76*logT-(805.1512535980252/T));
  Pr = S_tbc[49]*k0/k_f[49];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[13])/(1.0+f1*f1));

  pmod[13] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[49] = k_f[49]*pmod[13];

  //  Troe Reaction #51
  Fcent[14] = (1.0 - (0.783))*exp(-T/(74.0)) + (0.783) *exp(-T/(2941.0)) + exp(-(6964.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[14]);
  N =   0.75 - 1.27*log10(Fcent[14]);
  k0 = exp(log(2.6200000000000006e+27)-4.76*logT-(1227.8556617369884/T));
  Pr = S_tbc[51]*k0/k_f[51];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[14])/(1.0+f1*f1));

  pmod[14] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[51] = k_f[51]*pmod[14];

  //  Troe Reaction #53
  Fcent[15] = (1.0 - (0.7824))*exp(-T/(271.0)) + (0.7824) *exp(-T/(2755.0)) + exp(-(6570.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[15]);
  N =   0.75 - 1.27*log10(Fcent[15]);
  k0 = exp(log(2.4700000000000005e+18)-2.57*logT-(213.86830173697544/T));
  Pr = S_tbc[53]*k0/k_f[53];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[15])/(1.0+f1*f1));

  pmod[15] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[53] = k_f[53]*pmod[15];

  //  Troe Reaction #55
  Fcent[16] = (1.0 - (0.7187))*exp(-T/(103.00000000000001)) + (0.7187) *exp(-T/(1291.0)) + exp(-(4160.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[16]);
  N =   0.75 - 1.27*log10(Fcent[16]);
  k0 = exp(log(1.2700000000000002e+26)-4.82*logT-(3286.0235537469403/T));
  Pr = S_tbc[55]*k0/k_f[55];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[16])/(1.0+f1*f1));

  pmod[16] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[55] = k_f[55]*pmod[16];

  //  Troe Reaction #56
  Fcent[17] = (1.0 - (0.758))*exp(-T/(94.0)) + (0.758) *exp(-T/(1555.0)) + exp(-(4200.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[17]);
  N =   0.75 - 1.27*log10(Fcent[17]);
  k0 = exp(log(2.2000000000000006e+24)-4.8*logT-(2797.9006062531375/T));
  Pr = S_tbc[56]*k0/k_f[56];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[17])/(1.0+f1*f1));

  pmod[17] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[56] = k_f[56]*pmod[17];

  //  Troe Reaction #58
  Fcent[18] = (1.0 - (0.6))*exp(-T/(100.0)) + (0.6) *exp(-T/(90000.0)) + exp(-(10000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[18]);
  N =   0.75 - 1.27*log10(Fcent[18]);
  k0 = exp(log(4.360000000000001e+25)-4.65*logT-(2556.35523017373/T));
  Pr = S_tbc[58]*k0/k_f[58];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[18])/(1.0+f1*f1));

  pmod[18] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[58] = k_f[58]*pmod[18];

  //  Troe Reaction #62
  Fcent[19] = (1.0 - (0.7))*exp(-T/(100.0)) + (0.7) *exp(-T/(90000.0)) + exp(-(10000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[19]);
  N =   0.75 - 1.27*log10(Fcent[19]);
  k0 = exp(log(4.660000000000001e+35)-7.44*logT-(7085.331031662621/T));
  Pr = S_tbc[62]*k0/k_f[62];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[19])/(1.0+f1*f1));

  pmod[19] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[62] = k_f[62]*pmod[19];

  //  Troe Reaction #69
  Fcent[20] = (1.0 - (0.6464))*exp(-T/(132.0)) + (0.6464) *exp(-T/(1315.0)) + exp(-(5566.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[20]);
  N =   0.75 - 1.27*log10(Fcent[20]);
  k0 = exp(log(3.750000000000001e+27)-4.8*logT-(956.117113647655/T));
  Pr = S_tbc[69]*k0/k_f[69];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[20])/(1.0+f1*f1));

  pmod[20] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[69] = k_f[69]*pmod[20];

  //  Troe Reaction #70
  Fcent[21] = (1.0 - (0.7507))*exp(-T/(98.50000000000001)) + (0.7507) *exp(-T/(1302.0)) + exp(-(4167.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[21]);
  N =   0.75 - 1.27*log10(Fcent[21]);
  k0 = exp(log(3.8000000000000006e+34)-7.27*logT-(3633.2450318610886/T));
  Pr = S_tbc[70]*k0/k_f[70];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[21])/(1.0+f1*f1));

  pmod[21] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[70] = k_f[70]*pmod[21];

  //  Troe Reaction #71
  Fcent[22] = (1.0 - (0.782))*exp(-T/(207.49999999999997)) + (0.782) *exp(-T/(2663.0)) + exp(-(6095.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[22]);
  N =   0.75 - 1.27*log10(Fcent[22]);
  k0 = exp(log(1.4000000000000004e+24)-3.86*logT-(1670.6888512159023/T));
  Pr = S_tbc[71]*k0/k_f[71];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[22])/(1.0+f1*f1));

  pmod[22] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[71] = k_f[71]*pmod[22];

  //  Troe Reaction #73
  Fcent[23] = (1.0 - (0.9753))*exp(-T/(209.99999999999997)) + (0.9753) *exp(-T/(983.9999999999999)) + exp(-(4374.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[23]);
  N =   0.75 - 1.27*log10(Fcent[23]);
  k0 = exp(log(6.0000000000000005e+35)-7.62*logT-(3507.440148486397/T));
  Pr = S_tbc[73]*k0/k_f[73];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[23])/(1.0+f1*f1));

  pmod[23] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[73] = k_f[73]*pmod[23];

  //  Troe Reaction #75
  Fcent[24] = (1.0 - (0.8422))*exp(-T/(125.0)) + (0.8422) *exp(-T/(2219.0)) + exp(-(6882.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[24]);
  N =   0.75 - 1.27*log10(Fcent[24]);
  k0 = exp(log(1.9900000000000005e+35)-7.08*logT-(3364.022581439249/T));
  Pr = S_tbc[75]*k0/k_f[75];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[24])/(1.0+f1*f1));

  pmod[24] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[75] = k_f[75]*pmod[24];

  //  Troe Reaction #82
  Fcent[25] = (1.0 - (0.932))*exp(-T/(197.00000000000003)) + (0.932) *exp(-T/(1540.0)) + exp(-(10300.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[25]);
  N =   0.75 - 1.27*log10(Fcent[25]);
  k0 = exp(log(5.07e+21)-3.42*logT-(42446.56765062089/T));
  Pr = S_tbc[82]*k0/k_f[82];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[25])/(1.0+f1*f1));

  pmod[25] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[82] = k_f[82]*pmod[25];

  //  Troe Reaction #84
  Fcent[26] = (1.0 - (0.7346))*exp(-T/(94.0)) + (0.7346) *exp(-T/(1756.0)) + exp(-(5182.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[26]);
  N =   0.75 - 1.27*log10(Fcent[26]);
  k0 = exp(log(2300000000000.0005)-0.9*logT-(-855.4732069479018/T));
  Pr = S_tbc[84]*k0/k_f[84];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[26])/(1.0+f1*f1));

  pmod[26] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[84] = k_f[84]*pmod[26];

  //  Troe Reaction #94
  Fcent[27] = (1.0 - (0.412))*exp(-T/(195.0)) + (0.412) *exp(-T/(5900.0)) + exp(-(6394.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[27]);
  N =   0.75 - 1.27*log10(Fcent[27]);
  k0 = exp(log(4.000000000000001e+30)-5.92*logT-(1580.1093351861243/T));
  Pr = S_tbc[94]*k0/k_f[94];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[27])/(1.0+f1*f1));

  pmod[27] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[94] = k_f[94]*pmod[27];

  //  Troe Reaction #130
  Fcent[28] = (1.0 - (0.5757))*exp(-T/(237.00000000000003)) + (0.5757) *exp(-T/(1652.0)) + exp(-(5069.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[28]);
  N =   0.75 - 1.27*log10(Fcent[28]);
  k0 = exp(log(2.6900000000000003e+22)-3.74*logT-(974.2330168536105/T));
  Pr = S_tbc[130]*k0/k_f[130];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[28])/(1.0+f1*f1));

  pmod[28] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[130] = k_f[130]*pmod[28];

  //  Troe Reaction #139
  Fcent[29] = (1.0 - (0.5907))*exp(-T/(275.0)) + (0.5907) *exp(-T/(1226.0)) + exp(-(5185.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[29]);
  N =   0.75 - 1.27*log10(Fcent[29]);
  k0 = exp(log(2.6900000000000006e+27)-5.11*logT-(3570.342590173743/T));
  Pr = S_tbc[139]*k0/k_f[139];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[29])/(1.0+f1*f1));

  pmod[29] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[139] = k_f[139]*pmod[29];

  //  Troe Reaction #146
  Fcent[30] = (1.0 - (0.6027))*exp(-T/(208.0)) + (0.6027) *exp(-T/(3921.9999999999995)) + exp(-(10180.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[30]);
  N =   0.75 - 1.27*log10(Fcent[30]);
  k0 = exp(log(1.88e+32)-6.36*logT-(2536.226448833779/T));
  Pr = S_tbc[146]*k0/k_f[146];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[30])/(1.0+f1*f1));

  pmod[30] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[146] = k_f[146]*pmod[30];

  //  Troe Reaction #157
  Fcent[31] = (1.0 - (0.619))*exp(-T/(73.2)) + (0.619) *exp(-T/(1180.0)) + exp(-(9999.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[31]);
  N =   0.75 - 1.27*log10(Fcent[31]);
  k0 = exp(log(3.400000000000001e+35)-7.03*logT-(1389.892351523591/T));
  Pr = S_tbc[157]*k0/k_f[157];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[31])/(1.0+f1*f1));

  pmod[31] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[157] = k_f[157]*pmod[31];

  //  Three Body Reaction #165
  //  Three Body Reaction #166
  //  Troe Reaction #173
  Fcent[34] = (1.0 - (0.7345))*exp(-T/(180.0)) + (0.7345) *exp(-T/(1035.0)) + exp(-(5417.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[34]);
  N =   0.75 - 1.27*log10(Fcent[34]);
  k0 = exp(log(1.5800000000000006e+48)-9.3*logT-(49214.870376179286/T));
  Pr = S_tbc[173]*k0/k_f[173];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[34])/(1.0+f1*f1));

  pmod[34] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[173] = k_f[173]*pmod[34];

  //  Lindeman Reaction #184
  Fcent[35] = 1.0;
  k0 = exp(log(637000000000.0001)-(28502.35437737009/T));
  Pr = S_tbc[184]*k0/k_f[184];
  pmod[35] = Pr/(1.0 + Pr);
  k_f[184] = k_f[184]*pmod[35];

  //  Three Body Reaction #186
  //  Three Body Reaction #204
  //  Three Body Reaction #211
  //  Three Body Reaction #226
  //  Three Body Reaction #229
  //  Lindeman Reaction #236
  Fcent[41] = 1.0;
  k0 = exp(log(1.4000000000000003e+20)-3.4*logT-(956.117113647655/T));
  Pr = S_tbc[236]*k0/k_f[236];
  pmod[41] = Pr/(1.0 + Pr);
  k_f[236] = k_f[236]*pmod[41];

  //  Troe Reaction #240
  Fcent[42] = (1.0 - (0.667))*exp(-T/(235.0)) + (0.667) *exp(-T/(2117.0)) + exp(-(4536.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[42]);
  N =   0.75 - 1.27*log10(Fcent[42]);
  k0 = exp(log(1.3000000000000002e+19)-3.16*logT-(372.38245478908664/T));
  Pr = S_tbc[240]*k0/k_f[240];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[42])/(1.0+f1*f1));

  pmod[42] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[240] = k_f[240]*pmod[42];

  //  Three Body Reaction #268
  //  Troe Reaction #288
  Fcent[44] = (1.0 - (0.578))*exp(-T/(122.0)) + (0.578) *exp(-T/(2535.0)) + exp(-(9365.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[44]);
  N =   0.75 - 1.27*log10(Fcent[44]);
  k0 = exp(log(4.820000000000001e+19)-2.8*logT-(296.8995247642718/T));
  Pr = S_tbc[288]*k0/k_f[288];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[44])/(1.0+f1*f1));

  pmod[44] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[288] = k_f[288]*pmod[44];

  //  Three Body Reaction #302
  //  Troe Reaction #303
  Fcent[46] = (1.0 - (0.465))*exp(-T/(201.0)) + (0.465) *exp(-T/(1772.9999999999998)) + exp(-(5333.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[46]);
  N =   0.75 - 1.27*log10(Fcent[46]);
  k0 = exp(log(1.0120000000000002e+36)-7.63*logT-(1939.4080821042432/T));
  Pr = S_tbc[303]*k0/k_f[303];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[46])/(1.0+f1*f1));

  pmod[46] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[303] = k_f[303]*pmod[46];

  //  Troe Reaction #311
  Fcent[47] = (1.0 - (0.1527))*exp(-T/(291.0)) + (0.1527) *exp(-T/(2742.0)) + exp(-(7748.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[47]);
  N =   0.75 - 1.27*log10(Fcent[47]);
  k0 = exp(log(2.7100000000000003e+68)-16.82*logT-(6574.563205161375/T));
  Pr = S_tbc[311]*k0/k_f[311];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[47])/(1.0+f1*f1));

  pmod[47] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[311] = k_f[311]*pmod[47];

  //  Troe Reaction #317
  Fcent[48] = (1.0 - (0.1894))*exp(-T/(277.0)) + (0.1894) *exp(-T/(8748.0)) + exp(-(7891.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[48]);
  N =   0.75 - 1.27*log10(Fcent[48]);
  k0 = exp(log(3.0000000000000007e+57)-14.6*logT-(9143.498923672574/T));
  Pr = S_tbc[317]*k0/k_f[317];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[48])/(1.0+f1*f1));

  pmod[48] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[317] = k_f[317]*pmod[48];

  //  Troe Reaction #319
  Fcent[49] = (1.0 - (0.315))*exp(-T/(369.0)) + (0.315) *exp(-T/(3284.9999999999995)) + exp(-(6667.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[49]);
  N =   0.75 - 1.27*log10(Fcent[49]);
  k0 = exp(log(4.420000000000001e+55)-13.545*logT-(5715.0642419454825/T));
  Pr = S_tbc[319]*k0/k_f[319];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[49])/(1.0+f1*f1));

  pmod[49] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[319] = k_f[319]*pmod[49];



  // ----------------------------------------------------------- >
  // Forward, backward, net rates of progress. ----------------- >
  // ----------------------------------------------------------- >

  q_f[0] =   S_tbc[0] * k_f[0] * pow(cs(2,id),2.0);
  q_b[0] = - S_tbc[0] * k_f[0]/K_c[0] * cs(3,id);
  q[  0] =   q_f[0] + q_b[0];

  q_f[1] =   S_tbc[1] * k_f[1] * cs(1,id) * cs(2,id);
  q_b[1] = - S_tbc[1] * k_f[1]/K_c[1] * cs(4,id);
  q[  1] =   q_f[1] + q_b[1];

  q_f[2] =   S_tbc[2] * k_f[2] * cs(0,id) * cs(2,id);
  q_b[2] = - S_tbc[2] * k_f[2]/K_c[2] * cs(1,id) * cs(4,id);
  q[  2] =   q_f[2] + q_b[2];

  q_f[3] =   S_tbc[3] * k_f[3] * cs(2,id) * cs(6,id);
  q_b[3] = - S_tbc[3] * k_f[3]/K_c[3] * cs(3,id) * cs(4,id);
  q[  3] =   q_f[3] + q_b[3];

  q_f[4] =   S_tbc[4] * k_f[4] * cs(2,id) * cs(7,id);
  q_b[4] = - S_tbc[4] * k_f[4]/K_c[4] * cs(4,id) * cs(6,id);
  q[  4] =   q_f[4] + q_b[4];

  q_f[5] =   S_tbc[5] * k_f[5] * cs(2,id) * cs(9,id);
  q_b[5] = - S_tbc[5] * k_f[5]/K_c[5] * cs(1,id) * cs(14,id);
  q[  5] =   q_f[5] + q_b[5];

  q_f[6] =   S_tbc[6] * k_f[6] * cs(2,id) * cs(10,id);
  q_b[6] = - S_tbc[6] * k_f[6]/K_c[6] * cs(1,id) * cs(16,id);
  q[  6] =   q_f[6] + q_b[6];

  q_f[7] =   S_tbc[7] * k_f[7] * cs(2,id) * cs(11,id);
  q_b[7] = - S_tbc[7] * k_f[7]/K_c[7] * cs(0,id) * cs(14,id);
  q[  7] =   q_f[7] + q_b[7];

  q_f[8] =   S_tbc[8] * k_f[8] * cs(2,id) * cs(11,id);
  q_b[8] = - S_tbc[8] * k_f[8]/K_c[8] * cs(1,id) * cs(16,id);
  q[  8] =   q_f[8] + q_b[8];

  q_f[9] =   S_tbc[9] * k_f[9] * cs(2,id) * cs(12,id);
  q_b[9] = - S_tbc[9] * k_f[9]/K_c[9] * cs(1,id) * cs(17,id);
  q[  9] =   q_f[9] + q_b[9];

  q_f[10] =   S_tbc[10] * k_f[10] * cs(2,id) * cs(13,id);
  q_b[10] = - S_tbc[10] * k_f[10]/K_c[10] * cs(4,id) * cs(12,id);
  q[  10] =   q_f[10] + q_b[10];

  q_f[11] =   k_f[11] * cs(2,id) * cs(14,id);
  q_b[11] = - k_f[11]/K_c[11] * cs(15,id);
  q[  11] =   q_f[11] + q_b[11];

  q_f[12] =   S_tbc[12] * k_f[12] * cs(2,id) * cs(16,id);
  q_b[12] = - S_tbc[12] * k_f[12]/K_c[12] * cs(4,id) * cs(14,id);
  q[  12] =   q_f[12] + q_b[12];

  q_f[13] =   S_tbc[13] * k_f[13] * cs(2,id) * cs(16,id);
  q_b[13] = - S_tbc[13] * k_f[13]/K_c[13] * cs(1,id) * cs(15,id);
  q[  13] =   q_f[13] + q_b[13];

  q_f[14] =   S_tbc[14] * k_f[14] * cs(2,id) * cs(17,id);
  q_b[14] = - S_tbc[14] * k_f[14]/K_c[14] * cs(4,id) * cs(16,id);
  q[  14] =   q_f[14] + q_b[14];

  q_f[15] =   S_tbc[15] * k_f[15] * cs(2,id) * cs(18,id);
  q_b[15] = - S_tbc[15] * k_f[15]/K_c[15] * cs(4,id) * cs(17,id);
  q[  15] =   q_f[15] + q_b[15];

  q_f[16] =   S_tbc[16] * k_f[16] * cs(2,id) * cs(19,id);
  q_b[16] = - S_tbc[16] * k_f[16]/K_c[16] * cs(4,id) * cs(17,id);
  q[  16] =   q_f[16] + q_b[16];

  q_f[17] =   S_tbc[17] * k_f[17] * cs(2,id) * cs(20,id);
  q_b[17] = - S_tbc[17] * k_f[17]/K_c[17] * cs(4,id) * cs(18,id);
  q[  17] =   q_f[17] + q_b[17];

  q_f[18] =   S_tbc[18] * k_f[18] * cs(2,id) * cs(20,id);
  q_b[18] = - S_tbc[18] * k_f[18]/K_c[18] * cs(4,id) * cs(19,id);
  q[  18] =   q_f[18] + q_b[18];

  q_f[19] =   S_tbc[19] * k_f[19] * cs(2,id) * cs(21,id);
  q_b[19] = - S_tbc[19] * k_f[19]/K_c[19] * cs(9,id) * cs(14,id);
  q[  19] =   q_f[19] + q_b[19];

  q_f[20] =   S_tbc[20] * k_f[20] * cs(2,id) * cs(22,id);
  q_b[20] = - S_tbc[20] * k_f[20]/K_c[20] * cs(1,id) * cs(27,id);
  q[  20] =   q_f[20] + q_b[20];

  q_f[21] =   S_tbc[21] * k_f[21] * cs(2,id) * cs(22,id);
  q_b[21] = - S_tbc[21] * k_f[21]/K_c[21] * cs(4,id) * cs(21,id);
  q[  21] =   q_f[21] + q_b[21];

  q_f[22] =   S_tbc[22] * k_f[22] * cs(2,id) * cs(22,id);
  q_b[22] = - S_tbc[22] * k_f[22]/K_c[22] * cs(10,id) * cs(14,id);
  q[  22] =   q_f[22] + q_b[22];

  q_f[23] =   S_tbc[23] * k_f[23] * cs(2,id) * cs(23,id);
  q_b[23] = - S_tbc[23] * k_f[23]/K_c[23] * cs(1,id) * cs(28,id);
  q[  23] =   q_f[23] + q_b[23];

  q_f[24] =   S_tbc[24] * k_f[24] * cs(2,id) * cs(24,id);
  q_b[24] = - S_tbc[24] * k_f[24]/K_c[24] * cs(12,id) * cs(16,id);
  q[  24] =   q_f[24] + q_b[24];

  q_f[25] =   S_tbc[25] * k_f[25] * cs(2,id) * cs(25,id);
  q_b[25] = - S_tbc[25] * k_f[25]/K_c[25] * cs(12,id) * cs(17,id);
  q[  25] =   q_f[25] + q_b[25];

  q_f[26] =   S_tbc[26] * k_f[26] * cs(2,id) * cs(26,id);
  q_b[26] = - S_tbc[26] * k_f[26]/K_c[26] * cs(4,id) * cs(25,id);
  q[  26] =   q_f[26] + q_b[26];

  q_f[27] =   S_tbc[27] * k_f[27] * cs(2,id) * cs(27,id);
  q_b[27] = - S_tbc[27] * k_f[27]/K_c[27] * cs(1,id) * pow(cs(14,id),2.0);
  q[  27] =   q_f[27] + q_b[27];

  q_f[28] =   S_tbc[28] * k_f[28] * cs(2,id) * cs(28,id);
  q_b[28] = - S_tbc[28] * k_f[28]/K_c[28] * cs(4,id) * cs(27,id);
  q[  28] =   q_f[28] + q_b[28];

  q_f[29] =   S_tbc[29] * k_f[29] * cs(2,id) * cs(28,id);
  q_b[29] = - S_tbc[29] * k_f[29]/K_c[29] * cs(10,id) * cs(15,id);
  q[  29] =   q_f[29] + q_b[29];

  q_f[30] =   S_tbc[30] * k_f[30] * cs(3,id) * cs(14,id);
  q_b[30] = - S_tbc[30] * k_f[30]/K_c[30] * cs(2,id) * cs(15,id);
  q[  30] =   q_f[30] + q_b[30];

  q_f[31] =   S_tbc[31] * k_f[31] * cs(3,id) * cs(17,id);
  q_b[31] = - S_tbc[31] * k_f[31]/K_c[31] * cs(6,id) * cs(16,id);
  q[  31] =   q_f[31] + q_b[31];

  q_f[32] =   S_tbc[32] * k_f[32] * cs(1,id) * cs(3,id);
  q_b[32] = - S_tbc[32] * k_f[32]/K_c[32] * cs(6,id);
  q[  32] =   q_f[32] + q_b[32];

  q_f[33] =   S_tbc[33] * k_f[33] * cs(1,id) * cs(3,id);
  q_b[33] = - S_tbc[33] * k_f[33]/K_c[33] * cs(6,id);
  q[  33] =   q_f[33] + q_b[33];

  q_f[34] =   S_tbc[34] * k_f[34] * cs(1,id) * cs(3,id);
  q_b[34] = - S_tbc[34] * k_f[34]/K_c[34] * cs(6,id);
  q[  34] =   q_f[34] + q_b[34];

  q_f[35] =   S_tbc[35] * k_f[35] * cs(1,id) * cs(3,id);
  q_b[35] = - S_tbc[35] * k_f[35]/K_c[35] * cs(6,id);
  q[  35] =   q_f[35] + q_b[35];

  q_f[36] =   S_tbc[36] * k_f[36] * cs(1,id) * cs(3,id);
  q_b[36] = - S_tbc[36] * k_f[36]/K_c[36] * cs(6,id);
  q[  36] =   q_f[36] + q_b[36];

  q_f[37] =   S_tbc[37] * k_f[37] * cs(1,id) * cs(3,id);
  q_b[37] = - S_tbc[37] * k_f[37]/K_c[37] * cs(2,id) * cs(4,id);
  q[  37] =   q_f[37] + q_b[37];

  q_f[38] =   S_tbc[38] * k_f[38] * pow(cs(1,id),2.0);
  q_b[38] = - S_tbc[38] * k_f[38]/K_c[38] * cs(0,id);
  q[  38] =   q_f[38] + q_b[38];

  q_f[39] =   S_tbc[39] * k_f[39] * pow(cs(1,id),2.0);
  q_b[39] = - S_tbc[39] * k_f[39]/K_c[39] * cs(0,id);
  q[  39] =   q_f[39] + q_b[39];

  q_f[40] =   S_tbc[40] * k_f[40] * pow(cs(1,id),2.0);
  q_b[40] = - S_tbc[40] * k_f[40]/K_c[40] * cs(0,id);
  q[  40] =   q_f[40] + q_b[40];

  q_f[41] =   S_tbc[41] * k_f[41] * pow(cs(1,id),2.0);
  q_b[41] = - S_tbc[41] * k_f[41]/K_c[41] * cs(0,id);
  q[  41] =   q_f[41] + q_b[41];

  q_f[42] =   S_tbc[42] * k_f[42] * cs(1,id) * cs(4,id);
  q_b[42] = - S_tbc[42] * k_f[42]/K_c[42] * cs(5,id);
  q[  42] =   q_f[42] + q_b[42];

  q_f[43] =   S_tbc[43] * k_f[43] * cs(1,id) * cs(6,id);
  q_b[43] = - S_tbc[43] * k_f[43]/K_c[43] * cs(2,id) * cs(5,id);
  q[  43] =   q_f[43] + q_b[43];

  q_f[44] =   S_tbc[44] * k_f[44] * cs(1,id) * cs(6,id);
  q_b[44] = - S_tbc[44] * k_f[44]/K_c[44] * cs(0,id) * cs(3,id);
  q[  44] =   q_f[44] + q_b[44];

  q_f[45] =   S_tbc[45] * k_f[45] * cs(1,id) * cs(6,id);
  q_b[45] = - S_tbc[45] * k_f[45]/K_c[45] * pow(cs(4,id),2.0);
  q[  45] =   q_f[45] + q_b[45];

  q_f[46] =   S_tbc[46] * k_f[46] * cs(1,id) * cs(7,id);
  q_b[46] = - S_tbc[46] * k_f[46]/K_c[46] * cs(0,id) * cs(6,id);
  q[  46] =   q_f[46] + q_b[46];

  q_f[47] =   S_tbc[47] * k_f[47] * cs(1,id) * cs(7,id);
  q_b[47] = - S_tbc[47] * k_f[47]/K_c[47] * cs(4,id) * cs(5,id);
  q[  47] =   q_f[47] + q_b[47];

  q_f[48] =   S_tbc[48] * k_f[48] * cs(1,id) * cs(9,id);
  q_b[48] = - S_tbc[48] * k_f[48]/K_c[48] * cs(0,id) * cs(8,id);
  q[  48] =   q_f[48] + q_b[48];

  q_f[49] =   k_f[49] * cs(1,id) * cs(10,id);
  q_b[49] = - k_f[49]/K_c[49] * cs(12,id);
  q[  49] =   q_f[49] + q_b[49];

  q_f[50] =   S_tbc[50] * k_f[50] * cs(1,id) * cs(11,id);
  q_b[50] = - S_tbc[50] * k_f[50]/K_c[50] * cs(0,id) * cs(9,id);
  q[  50] =   q_f[50] + q_b[50];

  q_f[51] =   k_f[51] * cs(1,id) * cs(12,id);
  q_b[51] = - k_f[51]/K_c[51] * cs(13,id);
  q[  51] =   q_f[51] + q_b[51];

  q_f[52] =   S_tbc[52] * k_f[52] * cs(1,id) * cs(13,id);
  q_b[52] = - S_tbc[52] * k_f[52]/K_c[52] * cs(0,id) * cs(12,id);
  q[  52] =   q_f[52] + q_b[52];

  q_f[53] =   k_f[53] * cs(1,id) * cs(16,id);
  q_b[53] = - k_f[53]/K_c[53] * cs(17,id);
  q[  53] =   q_f[53] + q_b[53];

  q_f[54] =   S_tbc[54] * k_f[54] * cs(1,id) * cs(16,id);
  q_b[54] = - S_tbc[54] * k_f[54]/K_c[54] * cs(0,id) * cs(14,id);
  q[  54] =   q_f[54] + q_b[54];

  q_f[55] =   k_f[55] * cs(1,id) * cs(17,id);
  q_b[55] = - k_f[55]/K_c[55] * cs(18,id);
  q[  55] =   q_f[55] + q_b[55];

  q_f[56] =   k_f[56] * cs(1,id) * cs(17,id);
  q_b[56] = - k_f[56]/K_c[56] * cs(19,id);
  q[  56] =   q_f[56] + q_b[56];

  q_f[57] =   S_tbc[57] * k_f[57] * cs(1,id) * cs(17,id);
  q_b[57] = - S_tbc[57] * k_f[57]/K_c[57] * cs(0,id) * cs(16,id);
  q[  57] =   q_f[57] + q_b[57];

  q_f[58] =   k_f[58] * cs(1,id) * cs(18,id);
  q_b[58] = - k_f[58]/K_c[58] * cs(20,id);
  q[  58] =   q_f[58] + q_b[58];

  q_f[59] =   S_tbc[59] * k_f[59] * cs(1,id) * cs(18,id);
  q_b[59] = - S_tbc[59] * k_f[59]/K_c[59] * cs(0,id) * cs(17,id);
  q[  59] =   q_f[59] + q_b[59];

  q_f[60] =   S_tbc[60] * k_f[60] * cs(1,id) * cs(18,id);
  q_b[60] = - S_tbc[60] * k_f[60]/K_c[60] * cs(4,id) * cs(12,id);
  q[  60] =   q_f[60] + q_b[60];

  q_f[61] =   S_tbc[61] * k_f[61] * cs(1,id) * cs(18,id);
  q_b[61] = - S_tbc[61] * k_f[61]/K_c[61] * cs(5,id) * cs(11,id);
  q[  61] =   q_f[61] + q_b[61];

  q_f[62] =   k_f[62] * cs(1,id) * cs(19,id);
  q_b[62] = - k_f[62]/K_c[62] * cs(20,id);
  q[  62] =   q_f[62] + q_b[62];

  q_f[63] =   S_tbc[63] * k_f[63] * cs(1,id) * cs(19,id);
  q_b[63] = - S_tbc[63] * k_f[63]/K_c[63] * cs(1,id) * cs(18,id);
  q[  63] =   q_f[63] + q_b[63];

  q_f[64] =   S_tbc[64] * k_f[64] * cs(1,id) * cs(19,id);
  q_b[64] = - S_tbc[64] * k_f[64]/K_c[64] * cs(0,id) * cs(17,id);
  q[  64] =   q_f[64] + q_b[64];

  q_f[65] =   S_tbc[65] * k_f[65] * cs(1,id) * cs(19,id);
  q_b[65] = - S_tbc[65] * k_f[65]/K_c[65] * cs(4,id) * cs(12,id);
  q[  65] =   q_f[65] + q_b[65];

  q_f[66] =   S_tbc[66] * k_f[66] * cs(1,id) * cs(19,id);
  q_b[66] = - S_tbc[66] * k_f[66]/K_c[66] * cs(5,id) * cs(11,id);
  q[  66] =   q_f[66] + q_b[66];

  q_f[67] =   S_tbc[67] * k_f[67] * cs(1,id) * cs(20,id);
  q_b[67] = - S_tbc[67] * k_f[67]/K_c[67] * cs(0,id) * cs(18,id);
  q[  67] =   q_f[67] + q_b[67];

  q_f[68] =   S_tbc[68] * k_f[68] * cs(1,id) * cs(20,id);
  q_b[68] = - S_tbc[68] * k_f[68]/K_c[68] * cs(0,id) * cs(19,id);
  q[  68] =   q_f[68] + q_b[68];

  q_f[69] =   k_f[69] * cs(1,id) * cs(21,id);
  q_b[69] = - k_f[69]/K_c[69] * cs(22,id);
  q[  69] =   q_f[69] + q_b[69];

  q_f[70] =   k_f[70] * cs(1,id) * cs(22,id);
  q_b[70] = - k_f[70]/K_c[70] * cs(23,id);
  q[  70] =   q_f[70] + q_b[70];

  q_f[71] =   k_f[71] * cs(1,id) * cs(23,id);
  q_b[71] = - k_f[71]/K_c[71] * cs(24,id);
  q[  71] =   q_f[71] + q_b[71];

  q_f[72] =   S_tbc[72] * k_f[72] * cs(1,id) * cs(23,id);
  q_b[72] = - S_tbc[72] * k_f[72]/K_c[72] * cs(0,id) * cs(22,id);
  q[  72] =   q_f[72] + q_b[72];

  q_f[73] =   k_f[73] * cs(1,id) * cs(24,id);
  q_b[73] = - k_f[73]/K_c[73] * cs(25,id);
  q[  73] =   q_f[73] + q_b[73];

  q_f[74] =   S_tbc[74] * k_f[74] * cs(1,id) * cs(24,id);
  q_b[74] = - S_tbc[74] * k_f[74]/K_c[74] * cs(0,id) * cs(23,id);
  q[  74] =   q_f[74] + q_b[74];

  q_f[75] =   k_f[75] * cs(1,id) * cs(25,id);
  q_b[75] = - k_f[75]/K_c[75] * cs(26,id);
  q[  75] =   q_f[75] + q_b[75];

  q_f[76] =   S_tbc[76] * k_f[76] * cs(1,id) * cs(25,id);
  q_b[76] = - S_tbc[76] * k_f[76]/K_c[76] * cs(0,id) * cs(24,id);
  q[  76] =   q_f[76] + q_b[76];

  q_f[77] =   S_tbc[77] * k_f[77] * cs(1,id) * cs(26,id);
  q_b[77] = - S_tbc[77] * k_f[77]/K_c[77] * cs(0,id) * cs(25,id);
  q[  77] =   q_f[77] + q_b[77];

  q_f[78] =   S_tbc[78] * k_f[78] * cs(1,id) * cs(27,id);
  q_b[78] = - S_tbc[78] * k_f[78]/K_c[78] * cs(11,id) * cs(14,id);
  q[  78] =   q_f[78] + q_b[78];

  q_f[79] =   S_tbc[79] * k_f[79] * cs(1,id) * cs(28,id);
  q_b[79] = - S_tbc[79] * k_f[79]/K_c[79] * cs(0,id) * cs(27,id);
  q[  79] =   q_f[79] + q_b[79];

  q_f[80] =   S_tbc[80] * k_f[80] * cs(1,id) * cs(28,id);
  q_b[80] = - S_tbc[80] * k_f[80]/K_c[80] * cs(12,id) * cs(14,id);
  q[  80] =   q_f[80] + q_b[80];

  q_f[81] =   S_tbc[81] * k_f[81] * cs(1,id) * cs(29,id);
  q_b[81] = - S_tbc[81] * k_f[81]/K_c[81] * cs(1,id) * cs(28,id);
  q[  81] =   q_f[81] + q_b[81];

  q_f[82] =   k_f[82] * cs(0,id) * cs(14,id);
  q_b[82] = - k_f[82]/K_c[82] * cs(17,id);
  q[  82] =   q_f[82] + q_b[82];

  q_f[83] =   S_tbc[83] * k_f[83] * cs(0,id) * cs(4,id);
  q_b[83] = - S_tbc[83] * k_f[83]/K_c[83] * cs(1,id) * cs(5,id);
  q[  83] =   q_f[83] + q_b[83];

  q_f[84] =   k_f[84] * pow(cs(4,id),2.0);
  q_b[84] = - k_f[84]/K_c[84] * cs(7,id);
  q[  84] =   q_f[84] + q_b[84];

  q_f[85] =   S_tbc[85] * k_f[85] * pow(cs(4,id),2.0);
  q_b[85] = - S_tbc[85] * k_f[85]/K_c[85] * cs(2,id) * cs(5,id);
  q[  85] =   q_f[85] + q_b[85];

  q_f[86] =   S_tbc[86] * k_f[86] * cs(4,id) * cs(6,id);
  q_b[86] = - S_tbc[86] * k_f[86]/K_c[86] * cs(3,id) * cs(5,id);
  q[  86] =   q_f[86] + q_b[86];

  q_f[87] =   S_tbc[87] * k_f[87] * cs(4,id) * cs(7,id);
  q_b[87] = - S_tbc[87] * k_f[87]/K_c[87] * cs(5,id) * cs(6,id);
  q[  87] =   q_f[87] + q_b[87];

  q_f[88] =   S_tbc[88] * k_f[88] * cs(4,id) * cs(7,id);
  q_b[88] = - S_tbc[88] * k_f[88]/K_c[88] * cs(5,id) * cs(6,id);
  q[  88] =   q_f[88] + q_b[88];

  q_f[89] =   S_tbc[89] * k_f[89] * cs(4,id) * cs(8,id);
  q_b[89] = - S_tbc[89] * k_f[89]/K_c[89] * cs(1,id) * cs(14,id);
  q[  89] =   q_f[89] + q_b[89];

  q_f[90] =   S_tbc[90] * k_f[90] * cs(4,id) * cs(9,id);
  q_b[90] = - S_tbc[90] * k_f[90]/K_c[90] * cs(1,id) * cs(16,id);
  q[  90] =   q_f[90] + q_b[90];

  q_f[91] =   S_tbc[91] * k_f[91] * cs(4,id) * cs(10,id);
  q_b[91] = - S_tbc[91] * k_f[91]/K_c[91] * cs(1,id) * cs(17,id);
  q[  91] =   q_f[91] + q_b[91];

  q_f[92] =   S_tbc[92] * k_f[92] * cs(4,id) * cs(10,id);
  q_b[92] = - S_tbc[92] * k_f[92]/K_c[92] * cs(5,id) * cs(9,id);
  q[  92] =   q_f[92] + q_b[92];

  q_f[93] =   S_tbc[93] * k_f[93] * cs(4,id) * cs(11,id);
  q_b[93] = - S_tbc[93] * k_f[93]/K_c[93] * cs(1,id) * cs(17,id);
  q[  93] =   q_f[93] + q_b[93];

  q_f[94] =   k_f[94] * cs(4,id) * cs(12,id);
  q_b[94] = - k_f[94]/K_c[94] * cs(20,id);
  q[  94] =   q_f[94] + q_b[94];

  q_f[95] =   S_tbc[95] * k_f[95] * cs(4,id) * cs(12,id);
  q_b[95] = - S_tbc[95] * k_f[95]/K_c[95] * cs(5,id) * cs(10,id);
  q[  95] =   q_f[95] + q_b[95];

  q_f[96] =   S_tbc[96] * k_f[96] * cs(4,id) * cs(12,id);
  q_b[96] = - S_tbc[96] * k_f[96]/K_c[96] * cs(5,id) * cs(11,id);
  q[  96] =   q_f[96] + q_b[96];

  q_f[97] =   S_tbc[97] * k_f[97] * cs(4,id) * cs(13,id);
  q_b[97] = - S_tbc[97] * k_f[97]/K_c[97] * cs(5,id) * cs(12,id);
  q[  97] =   q_f[97] + q_b[97];

  q_f[98] =   S_tbc[98] * k_f[98] * cs(4,id) * cs(14,id);
  q_b[98] = - S_tbc[98] * k_f[98]/K_c[98] * cs(1,id) * cs(15,id);
  q[  98] =   q_f[98] + q_b[98];

  q_f[99] =   S_tbc[99] * k_f[99] * cs(4,id) * cs(16,id);
  q_b[99] = - S_tbc[99] * k_f[99]/K_c[99] * cs(5,id) * cs(14,id);
  q[  99] =   q_f[99] + q_b[99];

  q_f[100] =   S_tbc[100] * k_f[100] * cs(4,id) * cs(17,id);
  q_b[100] = - S_tbc[100] * k_f[100]/K_c[100] * cs(5,id) * cs(16,id);
  q[  100] =   q_f[100] + q_b[100];

  q_f[101] =   S_tbc[101] * k_f[101] * cs(4,id) * cs(18,id);
  q_b[101] = - S_tbc[101] * k_f[101]/K_c[101] * cs(5,id) * cs(17,id);
  q[  101] =   q_f[101] + q_b[101];

  q_f[102] =   S_tbc[102] * k_f[102] * cs(4,id) * cs(19,id);
  q_b[102] = - S_tbc[102] * k_f[102]/K_c[102] * cs(5,id) * cs(17,id);
  q[  102] =   q_f[102] + q_b[102];

  q_f[103] =   S_tbc[103] * k_f[103] * cs(4,id) * cs(20,id);
  q_b[103] = - S_tbc[103] * k_f[103]/K_c[103] * cs(5,id) * cs(18,id);
  q[  103] =   q_f[103] + q_b[103];

  q_f[104] =   S_tbc[104] * k_f[104] * cs(4,id) * cs(20,id);
  q_b[104] = - S_tbc[104] * k_f[104]/K_c[104] * cs(5,id) * cs(19,id);
  q[  104] =   q_f[104] + q_b[104];

  q_f[105] =   S_tbc[105] * k_f[105] * cs(4,id) * cs(21,id);
  q_b[105] = - S_tbc[105] * k_f[105]/K_c[105] * cs(1,id) * cs(27,id);
  q[  105] =   q_f[105] + q_b[105];

  q_f[106] =   S_tbc[106] * k_f[106] * cs(4,id) * cs(22,id);
  q_b[106] = - S_tbc[106] * k_f[106]/K_c[106] * cs(1,id) * cs(28,id);
  q[  106] =   q_f[106] + q_b[106];

  q_f[107] =   S_tbc[107] * k_f[107] * cs(4,id) * cs(22,id);
  q_b[107] = - S_tbc[107] * k_f[107]/K_c[107] * cs(1,id) * cs(29,id);
  q[  107] =   q_f[107] + q_b[107];

  q_f[108] =   S_tbc[108] * k_f[108] * cs(4,id) * cs(22,id);
  q_b[108] = - S_tbc[108] * k_f[108]/K_c[108] * cs(5,id) * cs(21,id);
  q[  108] =   q_f[108] + q_b[108];

  q_f[109] =   S_tbc[109] * k_f[109] * cs(4,id) * cs(22,id);
  q_b[109] = - S_tbc[109] * k_f[109]/K_c[109] * cs(12,id) * cs(14,id);
  q[  109] =   q_f[109] + q_b[109];

  q_f[110] =   S_tbc[110] * k_f[110] * cs(4,id) * cs(23,id);
  q_b[110] = - S_tbc[110] * k_f[110]/K_c[110] * cs(5,id) * cs(22,id);
  q[  110] =   q_f[110] + q_b[110];

  q_f[111] =   S_tbc[111] * k_f[111] * cs(4,id) * cs(24,id);
  q_b[111] = - S_tbc[111] * k_f[111]/K_c[111] * cs(5,id) * cs(23,id);
  q[  111] =   q_f[111] + q_b[111];

  q_f[112] =   S_tbc[112] * k_f[112] * cs(4,id) * cs(26,id);
  q_b[112] = - S_tbc[112] * k_f[112]/K_c[112] * cs(5,id) * cs(25,id);
  q[  112] =   q_f[112] + q_b[112];

  q_f[113] =   S_tbc[113] * k_f[113] * cs(4,id) * cs(28,id);
  q_b[113] = - S_tbc[113] * k_f[113]/K_c[113] * cs(5,id) * cs(27,id);
  q[  113] =   q_f[113] + q_b[113];

  q_f[114] =   S_tbc[114] * k_f[114] * pow(cs(6,id),2.0);
  q_b[114] = - S_tbc[114] * k_f[114]/K_c[114] * cs(3,id) * cs(7,id);
  q[  114] =   q_f[114] + q_b[114];

  q_f[115] =   S_tbc[115] * k_f[115] * pow(cs(6,id),2.0);
  q_b[115] = - S_tbc[115] * k_f[115]/K_c[115] * cs(3,id) * cs(7,id);
  q[  115] =   q_f[115] + q_b[115];

  q_f[116] =   S_tbc[116] * k_f[116] * cs(6,id) * cs(10,id);
  q_b[116] = - S_tbc[116] * k_f[116]/K_c[116] * cs(4,id) * cs(17,id);
  q[  116] =   q_f[116] + q_b[116];

  q_f[117] =   S_tbc[117] * k_f[117] * cs(6,id) * cs(12,id);
  q_b[117] = - S_tbc[117] * k_f[117]/K_c[117] * cs(3,id) * cs(13,id);
  q[  117] =   q_f[117] + q_b[117];

  q_f[118] =   S_tbc[118] * k_f[118] * cs(6,id) * cs(12,id);
  q_b[118] = - S_tbc[118] * k_f[118]/K_c[118] * cs(4,id) * cs(19,id);
  q[  118] =   q_f[118] + q_b[118];

  q_f[119] =   S_tbc[119] * k_f[119] * cs(6,id) * cs(14,id);
  q_b[119] = - S_tbc[119] * k_f[119]/K_c[119] * cs(4,id) * cs(15,id);
  q[  119] =   q_f[119] + q_b[119];

  q_f[120] =   S_tbc[120] * k_f[120] * cs(6,id) * cs(17,id);
  q_b[120] = - S_tbc[120] * k_f[120]/K_c[120] * cs(7,id) * cs(16,id);
  q[  120] =   q_f[120] + q_b[120];

  q_f[121] =   S_tbc[121] * k_f[121] * cs(3,id) * cs(8,id);
  q_b[121] = - S_tbc[121] * k_f[121]/K_c[121] * cs(2,id) * cs(14,id);
  q[  121] =   q_f[121] + q_b[121];

  q_f[122] =   S_tbc[122] * k_f[122] * cs(8,id) * cs(10,id);
  q_b[122] = - S_tbc[122] * k_f[122]/K_c[122] * cs(1,id) * cs(21,id);
  q[  122] =   q_f[122] + q_b[122];

  q_f[123] =   S_tbc[123] * k_f[123] * cs(8,id) * cs(12,id);
  q_b[123] = - S_tbc[123] * k_f[123]/K_c[123] * cs(1,id) * cs(22,id);
  q[  123] =   q_f[123] + q_b[123];

  q_f[124] =   S_tbc[124] * k_f[124] * cs(3,id) * cs(9,id);
  q_b[124] = - S_tbc[124] * k_f[124]/K_c[124] * cs(2,id) * cs(16,id);
  q[  124] =   q_f[124] + q_b[124];

  q_f[125] =   S_tbc[125] * k_f[125] * cs(0,id) * cs(9,id);
  q_b[125] = - S_tbc[125] * k_f[125]/K_c[125] * cs(1,id) * cs(10,id);
  q[  125] =   q_f[125] + q_b[125];

  q_f[126] =   S_tbc[126] * k_f[126] * cs(5,id) * cs(9,id);
  q_b[126] = - S_tbc[126] * k_f[126]/K_c[126] * cs(1,id) * cs(17,id);
  q[  126] =   q_f[126] + q_b[126];

  q_f[127] =   S_tbc[127] * k_f[127] * cs(9,id) * cs(10,id);
  q_b[127] = - S_tbc[127] * k_f[127]/K_c[127] * cs(1,id) * cs(22,id);
  q[  127] =   q_f[127] + q_b[127];

  q_f[128] =   S_tbc[128] * k_f[128] * cs(9,id) * cs(12,id);
  q_b[128] = - S_tbc[128] * k_f[128]/K_c[128] * cs(1,id) * cs(23,id);
  q[  128] =   q_f[128] + q_b[128];

  q_f[129] =   S_tbc[129] * k_f[129] * cs(9,id) * cs(13,id);
  q_b[129] = - S_tbc[129] * k_f[129]/K_c[129] * cs(1,id) * cs(24,id);
  q[  129] =   q_f[129] + q_b[129];

  q_f[130] =   k_f[130] * cs(9,id) * cs(14,id);
  q_b[130] = - k_f[130]/K_c[130] * cs(27,id);
  q[  130] =   q_f[130] + q_b[130];

  q_f[131] =   S_tbc[131] * k_f[131] * cs(9,id) * cs(15,id);
  q_b[131] = - S_tbc[131] * k_f[131]/K_c[131] * cs(14,id) * cs(16,id);
  q[  131] =   q_f[131] + q_b[131];

  q_f[132] =   S_tbc[132] * k_f[132] * cs(9,id) * cs(17,id);
  q_b[132] = - S_tbc[132] * k_f[132]/K_c[132] * cs(1,id) * cs(28,id);
  q[  132] =   q_f[132] + q_b[132];

  q_f[133] =   S_tbc[133] * k_f[133] * cs(9,id) * cs(27,id);
  q_b[133] = - S_tbc[133] * k_f[133]/K_c[133] * cs(14,id) * cs(22,id);
  q[  133] =   q_f[133] + q_b[133];

  q_f[134] =   S_tbc[134] * k_f[134] * cs(3,id) * cs(10,id);
  q_b[134] = - S_tbc[134] * k_f[134]/K_c[134] * cs(1,id) * cs(4,id) * cs(14,id);
  q[  134] =   q_f[134];

  q_f[135] =   S_tbc[135] * k_f[135] * cs(0,id) * cs(10,id);
  q_b[135] = - S_tbc[135] * k_f[135]/K_c[135] * cs(1,id) * cs(12,id);
  q[  135] =   q_f[135] + q_b[135];

  q_f[136] =   S_tbc[136] * k_f[136] * pow(cs(10,id),2.0);
  q_b[136] = - S_tbc[136] * k_f[136]/K_c[136] * cs(0,id) * cs(22,id);
  q[  136] =   q_f[136] + q_b[136];

  q_f[137] =   S_tbc[137] * k_f[137] * cs(10,id) * cs(12,id);
  q_b[137] = - S_tbc[137] * k_f[137]/K_c[137] * cs(1,id) * cs(24,id);
  q[  137] =   q_f[137] + q_b[137];

  q_f[138] =   S_tbc[138] * k_f[138] * cs(10,id) * cs(13,id);
  q_b[138] = - S_tbc[138] * k_f[138]/K_c[138] * pow(cs(12,id),2.0);
  q[  138] =   q_f[138] + q_b[138];

  q_f[139] =   k_f[139] * cs(10,id) * cs(14,id);
  q_b[139] = - k_f[139]/K_c[139] * cs(28,id);
  q[  139] =   q_f[139] + q_b[139];

  q_f[140] =   S_tbc[140] * k_f[140] * cs(10,id) * cs(27,id);
  q_b[140] = - S_tbc[140] * k_f[140]/K_c[140] * cs(14,id) * cs(23,id);
  q[  140] =   q_f[140] + q_b[140];

  q_f[141] =   S_tbc[141] * k_f[141] * cs(11,id) * cs(47,id);
  q_b[141] = - S_tbc[141] * k_f[141]/K_c[141] * cs(10,id) * cs(47,id);
  q[  141] =   q_f[141] + q_b[141];

  q_f[142] =   S_tbc[142] * k_f[142] * cs(11,id) * cs(48,id);
  q_b[142] = - S_tbc[142] * k_f[142]/K_c[142] * cs(10,id) * cs(48,id);
  q[  142] =   q_f[142] + q_b[142];

  q_f[143] =   S_tbc[143] * k_f[143] * cs(3,id) * cs(11,id);
  q_b[143] = - S_tbc[143] * k_f[143]/K_c[143] * cs(1,id) * cs(4,id) * cs(14,id);
  q[  143] =   q_f[143] + q_b[143];

  q_f[144] =   S_tbc[144] * k_f[144] * cs(3,id) * cs(11,id);
  q_b[144] = - S_tbc[144] * k_f[144]/K_c[144] * cs(5,id) * cs(14,id);
  q[  144] =   q_f[144] + q_b[144];

  q_f[145] =   S_tbc[145] * k_f[145] * cs(0,id) * cs(11,id);
  q_b[145] = - S_tbc[145] * k_f[145]/K_c[145] * cs(1,id) * cs(12,id);
  q[  145] =   q_f[145] + q_b[145];

  q_f[146] =   k_f[146] * cs(5,id) * cs(11,id);
  q_b[146] = - k_f[146]/K_c[146] * cs(20,id);
  q[  146] =   q_f[146] + q_b[146];

  q_f[147] =   S_tbc[147] * k_f[147] * cs(5,id) * cs(11,id);
  q_b[147] = - S_tbc[147] * k_f[147]/K_c[147] * cs(5,id) * cs(10,id);
  q[  147] =   q_f[147] + q_b[147];

  q_f[148] =   S_tbc[148] * k_f[148] * cs(11,id) * cs(12,id);
  q_b[148] = - S_tbc[148] * k_f[148]/K_c[148] * cs(1,id) * cs(24,id);
  q[  148] =   q_f[148] + q_b[148];

  q_f[149] =   S_tbc[149] * k_f[149] * cs(11,id) * cs(13,id);
  q_b[149] = - S_tbc[149] * k_f[149]/K_c[149] * pow(cs(12,id),2.0);
  q[  149] =   q_f[149] + q_b[149];

  q_f[150] =   S_tbc[150] * k_f[150] * cs(11,id) * cs(14,id);
  q_b[150] = - S_tbc[150] * k_f[150]/K_c[150] * cs(10,id) * cs(14,id);
  q[  150] =   q_f[150] + q_b[150];

  q_f[151] =   S_tbc[151] * k_f[151] * cs(11,id) * cs(15,id);
  q_b[151] = - S_tbc[151] * k_f[151]/K_c[151] * cs(10,id) * cs(15,id);
  q[  151] =   q_f[151] + q_b[151];

  q_f[152] =   S_tbc[152] * k_f[152] * cs(11,id) * cs(15,id);
  q_b[152] = - S_tbc[152] * k_f[152]/K_c[152] * cs(14,id) * cs(17,id);
  q[  152] =   q_f[152] + q_b[152];

  q_f[153] =   S_tbc[153] * k_f[153] * cs(11,id) * cs(26,id);
  q_b[153] = - S_tbc[153] * k_f[153]/K_c[153] * cs(12,id) * cs(25,id);
  q[  153] =   q_f[153] + q_b[153];

  q_f[154] =   S_tbc[154] * k_f[154] * cs(3,id) * cs(12,id);
  q_b[154] = - S_tbc[154] * k_f[154]/K_c[154] * cs(2,id) * cs(19,id);
  q[  154] =   q_f[154] + q_b[154];

  q_f[155] =   S_tbc[155] * k_f[155] * cs(3,id) * cs(12,id);
  q_b[155] = - S_tbc[155] * k_f[155]/K_c[155] * cs(4,id) * cs(17,id);
  q[  155] =   q_f[155] + q_b[155];

  q_f[156] =   S_tbc[156] * k_f[156] * cs(7,id) * cs(12,id);
  q_b[156] = - S_tbc[156] * k_f[156]/K_c[156] * cs(6,id) * cs(13,id);
  q[  156] =   q_f[156] + q_b[156];

  q_f[157] =   k_f[157] * pow(cs(12,id),2.0);
  q_b[157] = - k_f[157]/K_c[157] * cs(26,id);
  q[  157] =   q_f[157] + q_b[157];

  q_f[158] =   S_tbc[158] * k_f[158] * pow(cs(12,id),2.0);
  q_b[158] = - S_tbc[158] * k_f[158]/K_c[158] * cs(1,id) * cs(25,id);
  q[  158] =   q_f[158] + q_b[158];

  q_f[159] =   S_tbc[159] * k_f[159] * cs(12,id) * cs(16,id);
  q_b[159] = - S_tbc[159] * k_f[159]/K_c[159] * cs(13,id) * cs(14,id);
  q[  159] =   q_f[159] + q_b[159];

  q_f[160] =   S_tbc[160] * k_f[160] * cs(12,id) * cs(17,id);
  q_b[160] = - S_tbc[160] * k_f[160]/K_c[160] * cs(13,id) * cs(16,id);
  q[  160] =   q_f[160] + q_b[160];

  q_f[161] =   S_tbc[161] * k_f[161] * cs(12,id) * cs(20,id);
  q_b[161] = - S_tbc[161] * k_f[161]/K_c[161] * cs(13,id) * cs(18,id);
  q[  161] =   q_f[161] + q_b[161];

  q_f[162] =   S_tbc[162] * k_f[162] * cs(12,id) * cs(20,id);
  q_b[162] = - S_tbc[162] * k_f[162]/K_c[162] * cs(13,id) * cs(19,id);
  q[  162] =   q_f[162] + q_b[162];

  q_f[163] =   S_tbc[163] * k_f[163] * cs(12,id) * cs(24,id);
  q_b[163] = - S_tbc[163] * k_f[163]/K_c[163] * cs(13,id) * cs(23,id);
  q[  163] =   q_f[163] + q_b[163];

  q_f[164] =   S_tbc[164] * k_f[164] * cs(12,id) * cs(26,id);
  q_b[164] = - S_tbc[164] * k_f[164]/K_c[164] * cs(13,id) * cs(25,id);
  q[  164] =   q_f[164] + q_b[164];

  q_f[165] =   S_tbc[165] * k_f[165] * cs(16,id);
  q_b[165] = - S_tbc[165] * k_f[165]/K_c[165] * cs(1,id) * cs(14,id);
  q[  165] =   q_f[165] + q_b[165];

  q_f[166] =   S_tbc[166] * k_f[166] * cs(16,id);
  q_b[166] = - S_tbc[166] * k_f[166]/K_c[166] * cs(1,id) * cs(14,id);
  q[  166] =   q_f[166] + q_b[166];

  q_f[167] =   S_tbc[167] * k_f[167] * cs(3,id) * cs(16,id);
  q_b[167] = - S_tbc[167] * k_f[167]/K_c[167] * cs(6,id) * cs(14,id);
  q[  167] =   q_f[167] + q_b[167];

  q_f[168] =   S_tbc[168] * k_f[168] * cs(3,id) * cs(18,id);
  q_b[168] = - S_tbc[168] * k_f[168]/K_c[168] * cs(6,id) * cs(17,id);
  q[  168] =   q_f[168] + q_b[168];

  q_f[169] =   S_tbc[169] * k_f[169] * cs(3,id) * cs(19,id);
  q_b[169] = - S_tbc[169] * k_f[169]/K_c[169] * cs(6,id) * cs(17,id);
  q[  169] =   q_f[169] + q_b[169];

  q_f[170] =   S_tbc[170] * k_f[170] * cs(3,id) * cs(21,id);
  q_b[170] = - S_tbc[170] * k_f[170]/K_c[170] * cs(14,id) * cs(16,id);
  q[  170] =   q_f[170] + q_b[170];

  q_f[171] =   S_tbc[171] * k_f[171] * cs(0,id) * cs(21,id);
  q_b[171] = - S_tbc[171] * k_f[171]/K_c[171] * cs(1,id) * cs(22,id);
  q[  171] =   q_f[171] + q_b[171];

  q_f[172] =   S_tbc[172] * k_f[172] * cs(3,id) * cs(23,id);
  q_b[172] = - S_tbc[172] * k_f[172]/K_c[172] * cs(16,id) * cs(17,id);
  q[  172] =   q_f[172] + q_b[172];

  q_f[173] =   k_f[173] * cs(24,id);
  q_b[173] = - k_f[173]/K_c[173] * cs(0,id) * cs(22,id);
  q[  173] =   q_f[173] + q_b[173];

  q_f[174] =   S_tbc[174] * k_f[174] * cs(3,id) * cs(25,id);
  q_b[174] = - S_tbc[174] * k_f[174]/K_c[174] * cs(6,id) * cs(24,id);
  q[  174] =   q_f[174] + q_b[174];

  q_f[175] =   S_tbc[175] * k_f[175] * cs(3,id) * cs(27,id);
  q_b[175] = - S_tbc[175] * k_f[175]/K_c[175] * cs(4,id) * pow(cs(14,id),2.0);
  q[  175] =   q_f[175] + q_b[175];

  q_f[176] =   S_tbc[176] * k_f[176] * pow(cs(27,id),2.0);
  q_b[176] = - S_tbc[176] * k_f[176]/K_c[176] * pow(cs(14,id),2.0) * cs(22,id);
  q[  176] =   q_f[176] + q_b[176];

  q_f[177] =   S_tbc[177] * k_f[177] * cs(30,id) * cs(35,id);
  q_b[177] = - S_tbc[177] * k_f[177]/K_c[177] * cs(2,id) * cs(47,id);
  q[  177] =   q_f[177] + q_b[177];

  q_f[178] =   S_tbc[178] * k_f[178] * cs(3,id) * cs(30,id);
  q_b[178] = - S_tbc[178] * k_f[178]/K_c[178] * cs(2,id) * cs(35,id);
  q[  178] =   q_f[178] + q_b[178];

  q_f[179] =   S_tbc[179] * k_f[179] * cs(4,id) * cs(30,id);
  q_b[179] = - S_tbc[179] * k_f[179]/K_c[179] * cs(1,id) * cs(35,id);
  q[  179] =   q_f[179] + q_b[179];

  q_f[180] =   S_tbc[180] * k_f[180] * cs(2,id) * cs(37,id);
  q_b[180] = - S_tbc[180] * k_f[180]/K_c[180] * cs(3,id) * cs(47,id);
  q[  180] =   q_f[180] + q_b[180];

  q_f[181] =   S_tbc[181] * k_f[181] * cs(2,id) * cs(37,id);
  q_b[181] = - S_tbc[181] * k_f[181]/K_c[181] * pow(cs(35,id),2.0);
  q[  181] =   q_f[181] + q_b[181];

  q_f[182] =   S_tbc[182] * k_f[182] * cs(1,id) * cs(37,id);
  q_b[182] = - S_tbc[182] * k_f[182]/K_c[182] * cs(4,id) * cs(47,id);
  q[  182] =   q_f[182] + q_b[182];

  q_f[183] =   S_tbc[183] * k_f[183] * cs(4,id) * cs(37,id);
  q_b[183] = - S_tbc[183] * k_f[183]/K_c[183] * cs(6,id) * cs(47,id);
  q[  183] =   q_f[183] + q_b[183];

  q_f[184] =   k_f[184] * cs(37,id);
  q_b[184] = - k_f[184]/K_c[184] * cs(2,id) * cs(47,id);
  q[  184] =   q_f[184] + q_b[184];

  q_f[185] =   S_tbc[185] * k_f[185] * cs(6,id) * cs(35,id);
  q_b[185] = - S_tbc[185] * k_f[185]/K_c[185] * cs(4,id) * cs(36,id);
  q[  185] =   q_f[185] + q_b[185];

  q_f[186] =   S_tbc[186] * k_f[186] * cs(2,id) * cs(35,id);
  q_b[186] = - S_tbc[186] * k_f[186]/K_c[186] * cs(36,id);
  q[  186] =   q_f[186] + q_b[186];

  q_f[187] =   S_tbc[187] * k_f[187] * cs(2,id) * cs(36,id);
  q_b[187] = - S_tbc[187] * k_f[187]/K_c[187] * cs(3,id) * cs(35,id);
  q[  187] =   q_f[187] + q_b[187];

  q_f[188] =   S_tbc[188] * k_f[188] * cs(1,id) * cs(36,id);
  q_b[188] = - S_tbc[188] * k_f[188]/K_c[188] * cs(4,id) * cs(35,id);
  q[  188] =   q_f[188] + q_b[188];

  q_f[189] =   S_tbc[189] * k_f[189] * cs(2,id) * cs(31,id);
  q_b[189] = - S_tbc[189] * k_f[189]/K_c[189] * cs(1,id) * cs(35,id);
  q[  189] =   q_f[189] + q_b[189];

  q_f[190] =   S_tbc[190] * k_f[190] * cs(1,id) * cs(31,id);
  q_b[190] = - S_tbc[190] * k_f[190]/K_c[190] * cs(0,id) * cs(30,id);
  q[  190] =   q_f[190] + q_b[190];

  q_f[191] =   S_tbc[191] * k_f[191] * cs(4,id) * cs(31,id);
  q_b[191] = - S_tbc[191] * k_f[191]/K_c[191] * cs(1,id) * cs(38,id);
  q[  191] =   q_f[191] + q_b[191];

  q_f[192] =   S_tbc[192] * k_f[192] * cs(4,id) * cs(31,id);
  q_b[192] = - S_tbc[192] * k_f[192]/K_c[192] * cs(5,id) * cs(30,id);
  q[  192] =   q_f[192] + q_b[192];

  q_f[193] =   S_tbc[193] * k_f[193] * cs(3,id) * cs(31,id);
  q_b[193] = - S_tbc[193] * k_f[193]/K_c[193] * cs(2,id) * cs(38,id);
  q[  193] =   q_f[193] + q_b[193];

  q_f[194] =   S_tbc[194] * k_f[194] * cs(3,id) * cs(31,id);
  q_b[194] = - S_tbc[194] * k_f[194]/K_c[194] * cs(4,id) * cs(35,id);
  q[  194] =   q_f[194] + q_b[194];

  q_f[195] =   S_tbc[195] * k_f[195] * cs(30,id) * cs(31,id);
  q_b[195] = - S_tbc[195] * k_f[195]/K_c[195] * cs(1,id) * cs(47,id);
  q[  195] =   q_f[195] + q_b[195];

  q_f[196] =   S_tbc[196] * k_f[196] * cs(5,id) * cs(31,id);
  q_b[196] = - S_tbc[196] * k_f[196]/K_c[196] * cs(0,id) * cs(38,id);
  q[  196] =   q_f[196] + q_b[196];

  q_f[197] =   S_tbc[197] * k_f[197] * cs(31,id) * cs(35,id);
  q_b[197] = - S_tbc[197] * k_f[197]/K_c[197] * cs(4,id) * cs(47,id);
  q[  197] =   q_f[197] + q_b[197];

  q_f[198] =   S_tbc[198] * k_f[198] * cs(31,id) * cs(35,id);
  q_b[198] = - S_tbc[198] * k_f[198]/K_c[198] * cs(1,id) * cs(37,id);
  q[  198] =   q_f[198] + q_b[198];

  q_f[199] =   S_tbc[199] * k_f[199] * cs(2,id) * cs(32,id);
  q_b[199] = - S_tbc[199] * k_f[199]/K_c[199] * cs(4,id) * cs(31,id);
  q[  199] =   q_f[199] + q_b[199];

  q_f[200] =   S_tbc[200] * k_f[200] * cs(2,id) * cs(32,id);
  q_b[200] = - S_tbc[200] * k_f[200]/K_c[200] * cs(1,id) * cs(38,id);
  q[  200] =   q_f[200] + q_b[200];

  q_f[201] =   S_tbc[201] * k_f[201] * cs(1,id) * cs(32,id);
  q_b[201] = - S_tbc[201] * k_f[201]/K_c[201] * cs(0,id) * cs(31,id);
  q[  201] =   q_f[201] + q_b[201];

  q_f[202] =   S_tbc[202] * k_f[202] * cs(4,id) * cs(32,id);
  q_b[202] = - S_tbc[202] * k_f[202]/K_c[202] * cs(5,id) * cs(31,id);
  q[  202] =   q_f[202] + q_b[202];

  q_f[203] =   S_tbc[203] * k_f[203] * cs(34,id);
  q_b[203] = - S_tbc[203] * k_f[203]/K_c[203] * cs(1,id) * cs(47,id);
  q[  203] =   q_f[203] + q_b[203];

  q_f[204] =   S_tbc[204] * k_f[204] * cs(34,id);
  q_b[204] = - S_tbc[204] * k_f[204]/K_c[204] * cs(1,id) * cs(47,id);
  q[  204] =   q_f[204] + q_b[204];

  q_f[205] =   S_tbc[205] * k_f[205] * cs(3,id) * cs(34,id);
  q_b[205] = - S_tbc[205] * k_f[205]/K_c[205] * cs(6,id) * cs(47,id);
  q[  205] =   q_f[205] + q_b[205];

  q_f[206] =   S_tbc[206] * k_f[206] * cs(2,id) * cs(34,id);
  q_b[206] = - S_tbc[206] * k_f[206]/K_c[206] * cs(4,id) * cs(47,id);
  q[  206] =   q_f[206] + q_b[206];

  q_f[207] =   S_tbc[207] * k_f[207] * cs(2,id) * cs(34,id);
  q_b[207] = - S_tbc[207] * k_f[207]/K_c[207] * cs(31,id) * cs(35,id);
  q[  207] =   q_f[207] + q_b[207];

  q_f[208] =   S_tbc[208] * k_f[208] * cs(1,id) * cs(34,id);
  q_b[208] = - S_tbc[208] * k_f[208]/K_c[208] * cs(0,id) * cs(47,id);
  q[  208] =   q_f[208] + q_b[208];

  q_f[209] =   S_tbc[209] * k_f[209] * cs(4,id) * cs(34,id);
  q_b[209] = - S_tbc[209] * k_f[209]/K_c[209] * cs(5,id) * cs(47,id);
  q[  209] =   q_f[209] + q_b[209];

  q_f[210] =   S_tbc[210] * k_f[210] * cs(12,id) * cs(34,id);
  q_b[210] = - S_tbc[210] * k_f[210]/K_c[210] * cs(13,id) * cs(47,id);
  q[  210] =   q_f[210] + q_b[210];

  q_f[211] =   S_tbc[211] * k_f[211] * cs(1,id) * cs(35,id);
  q_b[211] = - S_tbc[211] * k_f[211]/K_c[211] * cs(38,id);
  q[  211] =   q_f[211] + q_b[211];

  q_f[212] =   S_tbc[212] * k_f[212] * cs(2,id) * cs(38,id);
  q_b[212] = - S_tbc[212] * k_f[212]/K_c[212] * cs(4,id) * cs(35,id);
  q[  212] =   q_f[212] + q_b[212];

  q_f[213] =   S_tbc[213] * k_f[213] * cs(1,id) * cs(38,id);
  q_b[213] = - S_tbc[213] * k_f[213]/K_c[213] * cs(0,id) * cs(35,id);
  q[  213] =   q_f[213] + q_b[213];

  q_f[214] =   S_tbc[214] * k_f[214] * cs(4,id) * cs(38,id);
  q_b[214] = - S_tbc[214] * k_f[214]/K_c[214] * cs(5,id) * cs(35,id);
  q[  214] =   q_f[214] + q_b[214];

  q_f[215] =   S_tbc[215] * k_f[215] * cs(3,id) * cs(38,id);
  q_b[215] = - S_tbc[215] * k_f[215]/K_c[215] * cs(6,id) * cs(35,id);
  q[  215] =   q_f[215] + q_b[215];

  q_f[216] =   S_tbc[216] * k_f[216] * cs(2,id) * cs(39,id);
  q_b[216] = - S_tbc[216] * k_f[216]/K_c[216] * cs(14,id) * cs(30,id);
  q[  216] =   q_f[216] + q_b[216];

  q_f[217] =   S_tbc[217] * k_f[217] * cs(4,id) * cs(39,id);
  q_b[217] = - S_tbc[217] * k_f[217]/K_c[217] * cs(1,id) * cs(46,id);
  q[  217] =   q_f[217] + q_b[217];

  q_f[218] =   S_tbc[218] * k_f[218] * cs(5,id) * cs(39,id);
  q_b[218] = - S_tbc[218] * k_f[218]/K_c[218] * cs(4,id) * cs(40,id);
  q[  218] =   q_f[218] + q_b[218];

  q_f[219] =   S_tbc[219] * k_f[219] * cs(3,id) * cs(39,id);
  q_b[219] = - S_tbc[219] * k_f[219]/K_c[219] * cs(2,id) * cs(46,id);
  q[  219] =   q_f[219] + q_b[219];

  q_f[220] =   S_tbc[220] * k_f[220] * cs(0,id) * cs(39,id);
  q_b[220] = - S_tbc[220] * k_f[220]/K_c[220] * cs(1,id) * cs(40,id);
  q[  220] =   q_f[220] + q_b[220];

  q_f[221] =   S_tbc[221] * k_f[221] * cs(2,id) * cs(46,id);
  q_b[221] = - S_tbc[221] * k_f[221]/K_c[221] * cs(14,id) * cs(35,id);
  q[  221] =   q_f[221] + q_b[221];

  q_f[222] =   S_tbc[222] * k_f[222] * cs(1,id) * cs(46,id);
  q_b[222] = - S_tbc[222] * k_f[222]/K_c[222] * cs(14,id) * cs(31,id);
  q[  222] =   q_f[222] + q_b[222];

  q_f[223] =   S_tbc[223] * k_f[223] * cs(4,id) * cs(46,id);
  q_b[223] = - S_tbc[223] * k_f[223]/K_c[223] * cs(1,id) * cs(14,id) * cs(35,id);
  q[  223] =   q_f[223] + q_b[223];

  q_f[224] =   S_tbc[224] * k_f[224] * cs(30,id) * cs(46,id);
  q_b[224] = - S_tbc[224] * k_f[224]/K_c[224] * cs(14,id) * cs(47,id);
  q[  224] =   q_f[224] + q_b[224];

  q_f[225] =   S_tbc[225] * k_f[225] * cs(3,id) * cs(46,id);
  q_b[225] = - S_tbc[225] * k_f[225]/K_c[225] * cs(15,id) * cs(35,id);
  q[  225] =   q_f[225] + q_b[225];

  q_f[226] =   S_tbc[226] * k_f[226] * cs(46,id);
  q_b[226] = - S_tbc[226] * k_f[226]/K_c[226] * cs(14,id) * cs(30,id);
  q[  226] =   q_f[226] + q_b[226];

  q_f[227] =   S_tbc[227] * k_f[227] * cs(35,id) * cs(46,id);
  q_b[227] = - S_tbc[227] * k_f[227]/K_c[227] * cs(14,id) * cs(37,id);
  q[  227] =   q_f[227] + q_b[227];

  q_f[228] =   S_tbc[228] * k_f[228] * cs(35,id) * cs(46,id);
  q_b[228] = - S_tbc[228] * k_f[228]/K_c[228] * cs(15,id) * cs(47,id);
  q[  228] =   q_f[228] + q_b[228];

  q_f[229] =   S_tbc[229] * k_f[229] * cs(40,id);
  q_b[229] = - S_tbc[229] * k_f[229]/K_c[229] * cs(1,id) * cs(39,id);
  q[  229] =   q_f[229] + q_b[229];

  q_f[230] =   S_tbc[230] * k_f[230] * cs(2,id) * cs(40,id);
  q_b[230] = - S_tbc[230] * k_f[230]/K_c[230] * cs(1,id) * cs(46,id);
  q[  230] =   q_f[230] + q_b[230];

  q_f[231] =   S_tbc[231] * k_f[231] * cs(2,id) * cs(40,id);
  q_b[231] = - S_tbc[231] * k_f[231]/K_c[231] * cs(14,id) * cs(31,id);
  q[  231] =   q_f[231] + q_b[231];

  q_f[232] =   S_tbc[232] * k_f[232] * cs(2,id) * cs(40,id);
  q_b[232] = - S_tbc[232] * k_f[232]/K_c[232] * cs(4,id) * cs(39,id);
  q[  232] =   q_f[232] + q_b[232];

  q_f[233] =   S_tbc[233] * k_f[233] * cs(4,id) * cs(40,id);
  q_b[233] = - S_tbc[233] * k_f[233]/K_c[233] * cs(1,id) * cs(44,id);
  q[  233] =   q_f[233] + q_b[233];

  q_f[234] =   S_tbc[234] * k_f[234] * cs(4,id) * cs(40,id);
  q_b[234] = - S_tbc[234] * k_f[234]/K_c[234] * cs(1,id) * cs(45,id);
  q[  234] =   q_f[234] + q_b[234];

  q_f[235] =   S_tbc[235] * k_f[235] * cs(4,id) * cs(40,id);
  q_b[235] = - S_tbc[235] * k_f[235]/K_c[235] * cs(14,id) * cs(32,id);
  q[  235] =   q_f[235] + q_b[235];

  q_f[236] =   k_f[236] * cs(1,id) * cs(40,id);
  q_b[236] = - k_f[236]/K_c[236] * cs(41,id);
  q[  236] =   q_f[236] + q_b[236];

  q_f[237] =   S_tbc[237] * k_f[237] * cs(30,id) * cs(41,id);
  q_b[237] = - S_tbc[237] * k_f[237]/K_c[237] * cs(10,id) * cs(47,id);
  q[  237] =   q_f[237] + q_b[237];

  q_f[238] =   S_tbc[238] * k_f[238] * cs(8,id) * cs(47,id);
  q_b[238] = - S_tbc[238] * k_f[238]/K_c[238] * cs(30,id) * cs(39,id);
  q[  238] =   q_f[238] + q_b[238];

  q_f[239] =   S_tbc[239] * k_f[239] * cs(9,id) * cs(47,id);
  q_b[239] = - S_tbc[239] * k_f[239]/K_c[239] * cs(30,id) * cs(40,id);
  q[  239] =   q_f[239] + q_b[239];

  q_f[240] =   k_f[240] * cs(9,id) * cs(47,id);
  q_b[240] = - k_f[240]/K_c[240] * cs(42,id);
  q[  240] =   q_f[240] + q_b[240];

  q_f[241] =   S_tbc[241] * k_f[241] * cs(10,id) * cs(47,id);
  q_b[241] = - S_tbc[241] * k_f[241]/K_c[241] * cs(31,id) * cs(40,id);
  q[  241] =   q_f[241] + q_b[241];

  q_f[242] =   S_tbc[242] * k_f[242] * cs(11,id) * cs(47,id);
  q_b[242] = - S_tbc[242] * k_f[242]/K_c[242] * cs(31,id) * cs(40,id);
  q[  242] =   q_f[242] + q_b[242];

  q_f[243] =   S_tbc[243] * k_f[243] * cs(8,id) * cs(35,id);
  q_b[243] = - S_tbc[243] * k_f[243]/K_c[243] * cs(2,id) * cs(39,id);
  q[  243] =   q_f[243] + q_b[243];

  q_f[244] =   S_tbc[244] * k_f[244] * cs(8,id) * cs(35,id);
  q_b[244] = - S_tbc[244] * k_f[244]/K_c[244] * cs(14,id) * cs(30,id);
  q[  244] =   q_f[244] + q_b[244];

  q_f[245] =   S_tbc[245] * k_f[245] * cs(9,id) * cs(35,id);
  q_b[245] = - S_tbc[245] * k_f[245]/K_c[245] * cs(2,id) * cs(40,id);
  q[  245] =   q_f[245] + q_b[245];

  q_f[246] =   S_tbc[246] * k_f[246] * cs(9,id) * cs(35,id);
  q_b[246] = - S_tbc[246] * k_f[246]/K_c[246] * cs(1,id) * cs(46,id);
  q[  246] =   q_f[246] + q_b[246];

  q_f[247] =   S_tbc[247] * k_f[247] * cs(9,id) * cs(35,id);
  q_b[247] = - S_tbc[247] * k_f[247]/K_c[247] * cs(16,id) * cs(30,id);
  q[  247] =   q_f[247] + q_b[247];

  q_f[248] =   S_tbc[248] * k_f[248] * cs(10,id) * cs(35,id);
  q_b[248] = - S_tbc[248] * k_f[248]/K_c[248] * cs(1,id) * cs(45,id);
  q[  248] =   q_f[248] + q_b[248];

  q_f[249] =   S_tbc[249] * k_f[249] * cs(10,id) * cs(35,id);
  q_b[249] = - S_tbc[249] * k_f[249]/K_c[249] * cs(4,id) * cs(40,id);
  q[  249] =   q_f[249] + q_b[249];

  q_f[250] =   S_tbc[250] * k_f[250] * cs(10,id) * cs(35,id);
  q_b[250] = - S_tbc[250] * k_f[250]/K_c[250] * cs(1,id) * cs(43,id);
  q[  250] =   q_f[250] + q_b[250];

  q_f[251] =   S_tbc[251] * k_f[251] * cs(11,id) * cs(35,id);
  q_b[251] = - S_tbc[251] * k_f[251]/K_c[251] * cs(1,id) * cs(45,id);
  q[  251] =   q_f[251] + q_b[251];

  q_f[252] =   S_tbc[252] * k_f[252] * cs(11,id) * cs(35,id);
  q_b[252] = - S_tbc[252] * k_f[252]/K_c[252] * cs(4,id) * cs(40,id);
  q[  252] =   q_f[252] + q_b[252];

  q_f[253] =   S_tbc[253] * k_f[253] * cs(11,id) * cs(35,id);
  q_b[253] = - S_tbc[253] * k_f[253]/K_c[253] * cs(1,id) * cs(43,id);
  q[  253] =   q_f[253] + q_b[253];

  q_f[254] =   S_tbc[254] * k_f[254] * cs(12,id) * cs(35,id);
  q_b[254] = - S_tbc[254] * k_f[254]/K_c[254] * cs(5,id) * cs(40,id);
  q[  254] =   q_f[254] + q_b[254];

  q_f[255] =   S_tbc[255] * k_f[255] * cs(12,id) * cs(35,id);
  q_b[255] = - S_tbc[255] * k_f[255]/K_c[255] * cs(4,id) * cs(41,id);
  q[  255] =   q_f[255] + q_b[255];

  q_f[256] =   S_tbc[256] * k_f[256] * cs(2,id) * cs(42,id);
  q_b[256] = - S_tbc[256] * k_f[256]/K_c[256] * cs(1,id) * cs(14,id) * cs(47,id);
  q[  256] =   q_f[256] + q_b[256];

  q_f[257] =   S_tbc[257] * k_f[257] * cs(2,id) * cs(42,id);
  q_b[257] = - S_tbc[257] * k_f[257]/K_c[257] * cs(35,id) * cs(40,id);
  q[  257] =   q_f[257] + q_b[257];

  q_f[258] =   S_tbc[258] * k_f[258] * cs(3,id) * cs(42,id);
  q_b[258] = - S_tbc[258] * k_f[258]/K_c[258] * cs(2,id) * cs(16,id) * cs(47,id);
  q[  258] =   q_f[258] + q_b[258];

  q_f[259] =   S_tbc[259] * k_f[259] * cs(4,id) * cs(42,id);
  q_b[259] = - S_tbc[259] * k_f[259]/K_c[259] * cs(1,id) * cs(16,id) * cs(47,id);
  q[  259] =   q_f[259] + q_b[259];

  q_f[260] =   S_tbc[260] * k_f[260] * cs(1,id) * cs(42,id);
  q_b[260] = - S_tbc[260] * k_f[260]/K_c[260] * cs(10,id) * cs(47,id);
  q[  260] =   q_f[260] + q_b[260];

  q_f[261] =   S_tbc[261] * k_f[261] * cs(2,id) * cs(45,id);
  q_b[261] = - S_tbc[261] * k_f[261]/K_c[261] * cs(15,id) * cs(31,id);
  q[  261] =   q_f[261] + q_b[261];

  q_f[262] =   S_tbc[262] * k_f[262] * cs(2,id) * cs(45,id);
  q_b[262] = - S_tbc[262] * k_f[262]/K_c[262] * cs(14,id) * cs(38,id);
  q[  262] =   q_f[262] + q_b[262];

  q_f[263] =   S_tbc[263] * k_f[263] * cs(2,id) * cs(45,id);
  q_b[263] = - S_tbc[263] * k_f[263]/K_c[263] * cs(4,id) * cs(46,id);
  q[  263] =   q_f[263] + q_b[263];

  q_f[264] =   S_tbc[264] * k_f[264] * cs(1,id) * cs(45,id);
  q_b[264] = - S_tbc[264] * k_f[264]/K_c[264] * cs(14,id) * cs(32,id);
  q[  264] =   q_f[264] + q_b[264];

  q_f[265] =   S_tbc[265] * k_f[265] * cs(1,id) * cs(45,id);
  q_b[265] = - S_tbc[265] * k_f[265]/K_c[265] * cs(0,id) * cs(46,id);
  q[  265] =   q_f[265] + q_b[265];

  q_f[266] =   S_tbc[266] * k_f[266] * cs(4,id) * cs(45,id);
  q_b[266] = - S_tbc[266] * k_f[266]/K_c[266] * cs(5,id) * cs(46,id);
  q[  266] =   q_f[266] + q_b[266];

  q_f[267] =   S_tbc[267] * k_f[267] * cs(4,id) * cs(45,id);
  q_b[267] = - S_tbc[267] * k_f[267]/K_c[267] * cs(15,id) * cs(32,id);
  q[  267] =   q_f[267] + q_b[267];

  q_f[268] =   S_tbc[268] * k_f[268] * cs(45,id);
  q_b[268] = - S_tbc[268] * k_f[268]/K_c[268] * cs(14,id) * cs(31,id);
  q[  268] =   q_f[268] + q_b[268];

  q_f[269] =   S_tbc[269] * k_f[269] * cs(1,id) * cs(43,id);
  q_b[269] = - S_tbc[269] * k_f[269]/K_c[269] * cs(1,id) * cs(45,id);
  q[  269] =   q_f[269] + q_b[269];

  q_f[270] =   S_tbc[270] * k_f[270] * cs(1,id) * cs(43,id);
  q_b[270] = - S_tbc[270] * k_f[270]/K_c[270] * cs(4,id) * cs(40,id);
  q[  270] =   q_f[270] + q_b[270];

  q_f[271] =   S_tbc[271] * k_f[271] * cs(1,id) * cs(43,id);
  q_b[271] = - S_tbc[271] * k_f[271]/K_c[271] * cs(14,id) * cs(32,id);
  q[  271] =   q_f[271] + q_b[271];

  q_f[272] =   S_tbc[272] * k_f[272] * cs(1,id) * cs(44,id);
  q_b[272] = - S_tbc[272] * k_f[272]/K_c[272] * cs(1,id) * cs(45,id);
  q[  272] =   q_f[272] + q_b[272];

  q_f[273] =   S_tbc[273] * k_f[273] * cs(27,id) * cs(35,id);
  q_b[273] = - S_tbc[273] * k_f[273]/K_c[273] * cs(14,id) * cs(43,id);
  q[  273] =   q_f[273] + q_b[273];

  q_f[274] =   S_tbc[274] * k_f[274] * cs(12,id) * cs(30,id);
  q_b[274] = - S_tbc[274] * k_f[274]/K_c[274] * cs(1,id) * cs(41,id);
  q[  274] =   q_f[274] + q_b[274];

  q_f[275] =   S_tbc[275] * k_f[275] * cs(12,id) * cs(30,id);
  q_b[275] = - S_tbc[275] * k_f[275]/K_c[275] * cs(0,id) * cs(40,id);
  q[  275] =   q_f[275] + q_b[275];

  q_f[276] =   S_tbc[276] * k_f[276] * cs(1,id) * cs(33,id);
  q_b[276] = - S_tbc[276] * k_f[276]/K_c[276] * cs(0,id) * cs(32,id);
  q[  276] =   q_f[276] + q_b[276];

  q_f[277] =   S_tbc[277] * k_f[277] * cs(4,id) * cs(33,id);
  q_b[277] = - S_tbc[277] * k_f[277]/K_c[277] * cs(5,id) * cs(32,id);
  q[  277] =   q_f[277] + q_b[277];

  q_f[278] =   S_tbc[278] * k_f[278] * cs(2,id) * cs(33,id);
  q_b[278] = - S_tbc[278] * k_f[278]/K_c[278] * cs(4,id) * cs(32,id);
  q[  278] =   q_f[278] + q_b[278];

  q_f[279] =   S_tbc[279] * k_f[279] * cs(15,id) * cs(31,id);
  q_b[279] = - S_tbc[279] * k_f[279]/K_c[279] * cs(14,id) * cs(38,id);
  q[  279] =   q_f[279] + q_b[279];

  q_f[280] =   S_tbc[280] * k_f[280] * cs(36,id) * cs(39,id);
  q_b[280] = - S_tbc[280] * k_f[280]/K_c[280] * cs(35,id) * cs(46,id);
  q[  280] =   q_f[280] + q_b[280];

  q_f[281] =   S_tbc[281] * k_f[281] * cs(36,id) * cs(46,id);
  q_b[281] = - S_tbc[281] * k_f[281]/K_c[281] * cs(15,id) * cs(37,id);
  q[  281] =   q_f[281] + q_b[281];

  q_f[282] =   S_tbc[282] * k_f[282] * cs(15,id) * cs(30,id);
  q_b[282] = - S_tbc[282] * k_f[282]/K_c[282] * cs(14,id) * cs(35,id);
  q[  282] =   q_f[282] + q_b[282];

  q_f[283] =   S_tbc[283] * k_f[283] * cs(2,id) * cs(12,id);
  q_b[283] = - S_tbc[283] * k_f[283]/K_c[283] * cs(0,id) * cs(1,id) * cs(14,id);
  q[  283] =   q_f[283];

  q_f[284] =   S_tbc[284] * k_f[284] * cs(2,id) * cs(24,id);
  q_b[284] = - S_tbc[284] * k_f[284]/K_c[284] * cs(1,id) * cs(51,id);
  q[  284] =   q_f[284] + q_b[284];

  q_f[285] =   S_tbc[285] * k_f[285] * cs(2,id) * cs(25,id);
  q_b[285] = - S_tbc[285] * k_f[285]/K_c[285] * cs(1,id) * cs(52,id);
  q[  285] =   q_f[285] + q_b[285];

  q_f[286] =   S_tbc[286] * k_f[286] * cs(4,id) * cs(6,id);
  q_b[286] = - S_tbc[286] * k_f[286]/K_c[286] * cs(3,id) * cs(5,id);
  q[  286] =   q_f[286] + q_b[286];

  q_f[287] =   S_tbc[287] * k_f[287] * cs(4,id) * cs(12,id);
  q_b[287] = - S_tbc[287] * k_f[287]/K_c[287] * cs(0,id) * cs(17,id);
  q[  287] =   q_f[287];

  q_f[288] =   k_f[288] * cs(0,id) * cs(9,id);
  q_b[288] = - k_f[288]/K_c[288] * cs(12,id);
  q[  288] =   q_f[288] + q_b[288];

  q_f[289] =   S_tbc[289] * k_f[289] * cs(3,id) * cs(10,id);
  q_b[289] = - S_tbc[289] * k_f[289]/K_c[289] * pow(cs(1,id),2.0) * cs(15,id);
  q[  289] =   q_f[289];

  q_f[290] =   S_tbc[290] * k_f[290] * cs(3,id) * cs(10,id);
  q_b[290] = - S_tbc[290] * k_f[290]/K_c[290] * cs(2,id) * cs(17,id);
  q[  290] =   q_f[290] + q_b[290];

  q_f[291] =   S_tbc[291] * k_f[291] * pow(cs(10,id),2.0);
  q_b[291] = - S_tbc[291] * k_f[291]/K_c[291] * pow(cs(1,id),2.0) * cs(22,id);
  q[  291] =   q_f[291];

  q_f[292] =   S_tbc[292] * k_f[292] * cs(5,id) * cs(11,id);
  q_b[292] = - S_tbc[292] * k_f[292]/K_c[292] * cs(0,id) * cs(17,id);
  q[  292] =   q_f[292];

  q_f[293] =   S_tbc[293] * k_f[293] * cs(3,id) * cs(23,id);
  q_b[293] = - S_tbc[293] * k_f[293]/K_c[293] * cs(2,id) * cs(51,id);
  q[  293] =   q_f[293] + q_b[293];

  q_f[294] =   S_tbc[294] * k_f[294] * cs(3,id) * cs(23,id);
  q_b[294] = - S_tbc[294] * k_f[294]/K_c[294] * cs(6,id) * cs(22,id);
  q[  294] =   q_f[294] + q_b[294];

  q_f[295] =   S_tbc[295] * k_f[295] * cs(2,id) * cs(52,id);
  q_b[295] = - S_tbc[295] * k_f[295]/K_c[295] * cs(4,id) * cs(51,id);
  q[  295] =   q_f[295] + q_b[295];

  q_f[296] =   S_tbc[296] * k_f[296] * cs(2,id) * cs(52,id);
  q_b[296] = - S_tbc[296] * k_f[296]/K_c[296] * cs(4,id) * cs(12,id) * cs(14,id);
  q[  296] =   q_f[296];

  q_f[297] =   S_tbc[297] * k_f[297] * cs(3,id) * cs(52,id);
  q_b[297] = - S_tbc[297] * k_f[297]/K_c[297] * cs(6,id) * cs(12,id) * cs(14,id);
  q[  297] =   q_f[297];

  q_f[298] =   S_tbc[298] * k_f[298] * cs(1,id) * cs(52,id);
  q_b[298] = - S_tbc[298] * k_f[298]/K_c[298] * cs(0,id) * cs(51,id);
  q[  298] =   q_f[298] + q_b[298];

  q_f[299] =   S_tbc[299] * k_f[299] * cs(1,id) * cs(52,id);
  q_b[299] = - S_tbc[299] * k_f[299]/K_c[299] * cs(0,id) * cs(12,id) * cs(14,id);
  q[  299] =   q_f[299];

  q_f[300] =   S_tbc[300] * k_f[300] * cs(4,id) * cs(52,id);
  q_b[300] = - S_tbc[300] * k_f[300]/K_c[300] * cs(5,id) * cs(12,id) * cs(14,id);
  q[  300] =   q_f[300];

  q_f[301] =   S_tbc[301] * k_f[301] * cs(6,id) * cs(52,id);
  q_b[301] = - S_tbc[301] * k_f[301]/K_c[301] * cs(7,id) * cs(12,id) * cs(14,id);
  q[  301] =   q_f[301];

  q_f[302] =   S_tbc[302] * k_f[302] * cs(52,id);
  q_b[302] = - S_tbc[302] * k_f[302]/K_c[302] * cs(13,id) * cs(14,id);
  q[  302] =   q_f[302];

  q_f[303] =   k_f[303] * cs(1,id) * cs(28,id);
  q_b[303] = - k_f[303]/K_c[303] * cs(51,id);
  q[  303] =   q_f[303] + q_b[303];

  q_f[304] =   S_tbc[304] * k_f[304] * cs(2,id) * cs(51,id);
  q_b[304] = - S_tbc[304] * k_f[304]/K_c[304] * cs(1,id) * cs(10,id) * cs(15,id);
  q[  304] =   q_f[304];

  q_f[305] =   S_tbc[305] * k_f[305] * cs(3,id) * cs(51,id);
  q_b[305] = - S_tbc[305] * k_f[305]/K_c[305] * cs(4,id) * cs(14,id) * cs(17,id);
  q[  305] =   q_f[305];

  q_f[306] =   S_tbc[306] * k_f[306] * cs(3,id) * cs(51,id);
  q_b[306] = - S_tbc[306] * k_f[306]/K_c[306] * cs(4,id) * pow(cs(16,id),2.0);
  q[  306] =   q_f[306];

  q_f[307] =   S_tbc[307] * k_f[307] * cs(1,id) * cs(51,id);
  q_b[307] = - S_tbc[307] * k_f[307]/K_c[307] * cs(12,id) * cs(16,id);
  q[  307] =   q_f[307] + q_b[307];

  q_f[308] =   S_tbc[308] * k_f[308] * cs(1,id) * cs(51,id);
  q_b[308] = - S_tbc[308] * k_f[308]/K_c[308] * cs(0,id) * cs(28,id);
  q[  308] =   q_f[308] + q_b[308];

  q_f[309] =   S_tbc[309] * k_f[309] * cs(4,id) * cs(51,id);
  q_b[309] = - S_tbc[309] * k_f[309]/K_c[309] * cs(5,id) * cs(28,id);
  q[  309] =   q_f[309] + q_b[309];

  q_f[310] =   S_tbc[310] * k_f[310] * cs(4,id) * cs(51,id);
  q_b[310] = - S_tbc[310] * k_f[310]/K_c[310] * cs(16,id) * cs(18,id);
  q[  310] =   q_f[310] + q_b[310];

  q_f[311] =   k_f[311] * cs(12,id) * cs(25,id);
  q_b[311] = - k_f[311]/K_c[311] * cs(50,id);
  q[  311] =   q_f[311] + q_b[311];

  q_f[312] =   S_tbc[312] * k_f[312] * cs(2,id) * cs(50,id);
  q_b[312] = - S_tbc[312] * k_f[312]/K_c[312] * cs(4,id) * cs(49,id);
  q[  312] =   q_f[312] + q_b[312];

  q_f[313] =   S_tbc[313] * k_f[313] * cs(1,id) * cs(50,id);
  q_b[313] = - S_tbc[313] * k_f[313]/K_c[313] * cs(0,id) * cs(49,id);
  q[  313] =   q_f[313] + q_b[313];

  q_f[314] =   S_tbc[314] * k_f[314] * cs(4,id) * cs(50,id);
  q_b[314] = - S_tbc[314] * k_f[314]/K_c[314] * cs(5,id) * cs(49,id);
  q[  314] =   q_f[314] + q_b[314];

  q_f[315] =   S_tbc[315] * k_f[315] * cs(7,id) * cs(49,id);
  q_b[315] = - S_tbc[315] * k_f[315]/K_c[315] * cs(6,id) * cs(50,id);
  q[  315] =   q_f[315] + q_b[315];

  q_f[316] =   S_tbc[316] * k_f[316] * cs(12,id) * cs(50,id);
  q_b[316] = - S_tbc[316] * k_f[316]/K_c[316] * cs(13,id) * cs(49,id);
  q[  316] =   q_f[316] + q_b[316];

  q_f[317] =   k_f[317] * cs(12,id) * cs(24,id);
  q_b[317] = - k_f[317]/K_c[317] * cs(49,id);
  q[  317] =   q_f[317] + q_b[317];

  q_f[318] =   S_tbc[318] * k_f[318] * cs(2,id) * cs(49,id);
  q_b[318] = - S_tbc[318] * k_f[318]/K_c[318] * cs(17,id) * cs(25,id);
  q[  318] =   q_f[318] + q_b[318];

  q_f[319] =   k_f[319] * cs(1,id) * cs(49,id);
  q_b[319] = - k_f[319]/K_c[319] * cs(50,id);
  q[  319] =   q_f[319] + q_b[319];

  q_f[320] =   S_tbc[320] * k_f[320] * cs(1,id) * cs(49,id);
  q_b[320] = - S_tbc[320] * k_f[320]/K_c[320] * cs(12,id) * cs(25,id);
  q[  320] =   q_f[320] + q_b[320];

  q_f[321] =   S_tbc[321] * k_f[321] * cs(4,id) * cs(49,id);
  q_b[321] = - S_tbc[321] * k_f[321]/K_c[321] * cs(18,id) * cs(25,id);
  q[  321] =   q_f[321] + q_b[321];

  q_f[322] =   S_tbc[322] * k_f[322] * cs(6,id) * cs(49,id);
  q_b[322] = - S_tbc[322] * k_f[322]/K_c[322] * cs(3,id) * cs(50,id);
  q[  322] =   q_f[322] + q_b[322];

  q_f[323] =   S_tbc[323] * k_f[323] * cs(6,id) * cs(49,id);
  q_b[323] = - S_tbc[323] * k_f[323]/K_c[323] * cs(4,id) * cs(17,id) * cs(25,id);
  q[  323] =   q_f[323];

  q_f[324] =   S_tbc[324] * k_f[324] * cs(12,id) * cs(49,id);
  q_b[324] = - S_tbc[324] * k_f[324]/K_c[324] * pow(cs(25,id),2.0);
  q[  324] =   q_f[324] + q_b[324];

  // ----------------------------------------------------------- >
  // Source terms. --------------------------------------------- >
  // ----------------------------------------------------------- >

  b.omega(i,j,k,1) = th.MW[0] * ( -q[2] +q[7] +q[38] +q[39] +q[40] +q[41] +q[44] +q[46] +q[48] +q[50] +q[52] +q[54] +q[57] +q[59] +q[64] +q[67] +q[68] +q[72] +q[74] +q[76] +q[77] +q[79] -q[82] -q[83] -q[125] -q[135] +q[136] -q[145] -q[171] +q[173] +q[190] +q[196] +q[201] +q[208] +q[213] -q[220] +q[265] +q[275] +q[276] +q[283] +q[287] -q[288] +q[292] +q[298] +q[299] +q[308] +q[313]);
  b.omega(i,j,k,2) = th.MW[1] * ( -q[1] +q[2] +q[5] +q[6] +q[8] +q[9] +q[13] +q[20] +q[23] +q[27] -q[32] -q[33] -q[34] -q[35] -q[36] -q[37] -2.0*q[38] -2.0*q[39] -2.0*q[40] -2.0*q[41] -q[42] -q[43] -q[44] -q[45] -q[46] -q[47] -q[48] -q[49] -q[50] -q[51] -q[52] -q[53] -q[54] -q[55] -q[56] -q[57] -q[58] -q[59] -q[60] -q[61] -q[62] -q[64] -q[65] -q[66] -q[67] -q[68] -q[69] -q[70] -q[71] -q[72] -q[73] -q[74] -q[75] -q[76] -q[77] -q[78] -q[79] -q[80] +q[83] +q[89] +q[90] +q[91] +q[93] +q[98] +q[105] +q[106] +q[107] +q[122] +q[123] +q[125] +q[126] +q[127] +q[128] +q[129] +q[132] +q[134] +q[135] +q[137] +q[143] +q[145] +q[148] +q[158] +q[165] +q[166] +q[171] +q[179] -q[182] -q[188] +q[189] -q[190] +q[191] +q[195] +q[198] +q[200] -q[201] +q[203] +q[204] -q[208] -q[211] -q[213] +q[217] +q[220] -q[222] +q[223] +q[229] +q[230] +q[233] +q[234] -q[236] +q[246] +q[248] +q[250] +q[251] +q[253] +q[256] +q[259] -q[260] -q[264] -q[265] -q[270] -q[271] +q[274] -q[276] +q[283] +q[284] +q[285] +2.0*q[289] +2.0*q[291] -q[298] -q[299] -q[303] +q[304] -q[307] -q[308] -q[313] -q[319] -q[320]);
  b.omega(i,j,k,3) = th.MW[2] * ( -2.0*q[0] -q[1] -q[2] -q[3] -q[4] -q[5] -q[6] -q[7] -q[8] -q[9] -q[10] -q[11] -q[12] -q[13] -q[14] -q[15] -q[16] -q[17] -q[18] -q[19] -q[20] -q[21] -q[22] -q[23] -q[24] -q[25] -q[26] -q[27] -q[28] -q[29] +q[30] +q[37] +q[43] +q[85] +q[121] +q[124] +q[154] +q[177] +q[178] -q[180] -q[181] +q[184] -q[186] -q[187] -q[189] +q[193] -q[199] -q[200] -q[206] -q[207] -q[212] -q[216] +q[219] -q[221] -q[230] -q[231] -q[232] +q[243] +q[245] -q[256] -q[257] +q[258] -q[261] -q[262] -q[263] -q[278] -q[283] -q[284] -q[285] +q[290] +q[293] -q[295] -q[296] -q[304] -q[312] -q[318]);
  b.omega(i,j,k,4) = th.MW[3] * ( q[0] +q[3] -q[30] -q[31] -q[32] -q[33] -q[34] -q[35] -q[36] -q[37] +q[44] +q[86] +q[114] +q[115] +q[117] -q[121] -q[124] -q[134] -q[143] -q[144] -q[154] -q[155] -q[167] -q[168] -q[169] -q[170] -q[172] -q[174] -q[175] -q[178] +q[180] +q[187] -q[193] -q[194] -q[205] -q[215] -q[219] -q[225] -q[258] +q[286] -q[289] -q[290] -q[293] -q[294] -q[297] -q[305] -q[306] +q[322]);
  b.omega(i,j,k,5) = th.MW[4] * ( q[1] +q[2] +q[3] +q[4] +q[10] +q[12] +q[14] +q[15] +q[16] +q[17] +q[18] +q[21] +q[26] +q[28] +q[37] -q[42] +2.0*q[45] +q[47] +q[60] +q[65] -q[83] -2.0*q[84] -2.0*q[85] -q[86] -q[87] -q[88] -q[89] -q[90] -q[91] -q[92] -q[93] -q[94] -q[95] -q[96] -q[97] -q[98] -q[99] -q[100] -q[101] -q[102] -q[103] -q[104] -q[105] -q[106] -q[107] -q[108] -q[109] -q[110] -q[111] -q[112] -q[113] +q[116] +q[118] +q[119] +q[134] +q[143] +q[155] +q[175] -q[179] +q[182] -q[183] +q[185] +q[188] -q[191] -q[192] +q[194] +q[197] +q[199] -q[202] +q[206] -q[209] +q[212] -q[214] -q[217] +q[218] -q[223] +q[232] -q[233] -q[234] -q[235] +q[249] +q[252] +q[255] -q[259] +q[263] -q[266] -q[267] +q[270] -q[277] +q[278] -q[286] -q[287] +q[295] +q[296] -q[300] +q[305] +q[306] -q[309] -q[310] +q[312] -q[314] -q[321] +q[323]);
  b.omega(i,j,k,6) = th.MW[5] * ( q[42] +q[43] +q[47] +q[61] +q[66] +q[83] +q[85] +q[86] +q[87] +q[88] +q[92] +q[95] +q[96] +q[97] +q[99] +q[100] +q[101] +q[102] +q[103] +q[104] +q[108] +q[110] +q[111] +q[112] +q[113] -q[126] +q[144] -q[146] +q[192] -q[196] +q[202] +q[209] +q[214] -q[218] +q[254] +q[266] +q[277] +q[286] -q[292] +q[300] +q[309] +q[314]);
  b.omega(i,j,k,7) = th.MW[6] * ( -q[3] +q[4] +q[31] +q[32] +q[33] +q[34] +q[35] +q[36] -q[43] -q[44] -q[45] +q[46] -q[86] +q[87] +q[88] -2.0*q[114] -2.0*q[115] -q[116] -q[117] -q[118] -q[119] -q[120] +q[156] +q[167] +q[168] +q[169] +q[174] +q[183] -q[185] +q[205] +q[215] -q[286] +q[294] +q[297] -q[301] +q[315] -q[322] -q[323]);
  b.omega(i,j,k,8) = th.MW[7] * ( -q[4] -q[46] -q[47] +q[84] -q[87] -q[88] +q[114] +q[115] +q[120] -q[156] +q[301] -q[315]);
  b.omega(i,j,k,9) = th.MW[8] * ( q[48] -q[89] -q[121] -q[122] -q[123] -q[238] -q[243] -q[244]);
  b.omega(i,j,k,10) = th.MW[9] * ( -q[5] +q[19] -q[48] +q[50] -q[90] +q[92] -q[124] -q[125] -q[126] -q[127] -q[128] -q[129] -q[130] -q[131] -q[132] -q[133] -q[239] -q[240] -q[245] -q[246] -q[247] -q[288]);
  b.omega(i,j,k,11) = th.MW[10] * ( -q[6] +q[22] +q[29] -q[49] -q[91] -q[92] +q[95] -q[116] -q[122] +q[125] -q[127] -q[134] -q[135] -2.0*q[136] -q[137] -q[138] -q[139] -q[140] +q[141] +q[142] +q[147] +q[150] +q[151] +q[237] -q[241] -q[248] -q[249] -q[250] +q[260] -q[289] -q[290] -2.0*q[291] +q[304]);
  b.omega(i,j,k,12) = th.MW[11] * ( -q[7] -q[8] -q[50] +q[61] +q[66] +q[78] -q[93] +q[96] -q[141] -q[142] -q[143] -q[144] -q[145] -q[146] -q[147] -q[148] -q[149] -q[150] -q[151] -q[152] -q[153] -q[242] -q[251] -q[252] -q[253] -q[292]);
  b.omega(i,j,k,13) = th.MW[12] * ( -q[9] +q[10] +q[24] +q[25] +q[49] -q[51] +q[52] +q[60] +q[65] +q[80] -q[94] -q[95] -q[96] +q[97] +q[109] -q[117] -q[118] -q[123] -q[128] +q[135] -q[137] +2.0*q[138] +q[145] -q[148] +2.0*q[149] +q[153] -q[154] -q[155] -q[156] -2.0*q[157] -2.0*q[158] -q[159] -q[160] -q[161] -q[162] -q[163] -q[164] -q[210] -q[254] -q[255] -q[274] -q[275] -q[283] -q[287] +q[288] +q[296] +q[297] +q[299] +q[300] +q[301] +q[307] -q[311] -q[316] -q[317] +q[320] -q[324]);
  b.omega(i,j,k,14) = th.MW[13] * ( -q[10] +q[51] -q[52] -q[97] +q[117] -q[129] -q[138] -q[149] +q[156] +q[159] +q[160] +q[161] +q[162] +q[163] +q[164] +q[210] +q[302] +q[316]);
  b.omega(i,j,k,15) = th.MW[14] * ( q[5] +q[7] -q[11] +q[12] +q[19] +q[22] +2.0*q[27] -q[30] +q[54] +q[78] +q[80] -q[82] +q[89] -q[98] +q[99] +q[109] -q[119] +q[121] -q[130] +q[131] +q[133] +q[134] -q[139] +q[140] +q[143] +q[144] +q[152] +q[159] +q[165] +q[166] +q[167] +q[170] +2.0*q[175] +2.0*q[176] +q[216] +q[221] +q[222] +q[223] +q[224] +q[226] +q[227] +q[231] +q[235] +q[244] +q[256] +q[262] +q[264] +q[268] +q[271] +q[273] +q[279] +q[282] +q[283] +q[296] +q[297] +q[299] +q[300] +q[301] +q[302] +q[305]);
  b.omega(i,j,k,16) = th.MW[15] * ( q[11] +q[13] +q[29] +q[30] +q[98] +q[119] -q[131] -q[152] +q[225] +q[228] +q[261] +q[267] -q[279] +q[281] -q[282] +q[289] +q[304]);
  b.omega(i,j,k,17) = th.MW[16] * ( q[6] +q[8] -q[12] -q[13] +q[14] +q[24] +q[31] -q[53] -q[54] +q[57] +q[90] -q[99] +q[100] +q[120] +q[124] +q[131] -q[159] +q[160] -q[165] -q[166] -q[167] +q[170] +q[172] +q[247] +q[258] +q[259] +2.0*q[306] +q[307] +q[310]);
  b.omega(i,j,k,18) = th.MW[17] * ( q[9] -q[14] +q[15] +q[16] +q[25] -q[31] +q[53] -q[55] -q[56] -q[57] +q[59] +q[64] +q[82] +q[91] +q[93] -q[100] +q[101] +q[102] +q[116] -q[120] +q[126] -q[132] +q[152] +q[155] -q[160] +q[168] +q[169] +q[172] +q[287] +q[290] +q[292] +q[305] +q[318] +q[323]);
  b.omega(i,j,k,19) = th.MW[18] * ( -q[15] +q[17] +q[55] -q[58] -q[59] -q[60] -q[61] +q[63] +q[67] -q[101] +q[103] +q[161] -q[168] +q[310] +q[321]);
  b.omega(i,j,k,20) = th.MW[19] * ( -q[16] +q[18] +q[56] -q[62] -q[63] -q[64] -q[65] -q[66] +q[68] -q[102] +q[104] +q[118] +q[154] +q[162] -q[169]);
  b.omega(i,j,k,21) = th.MW[20] * ( -q[17] -q[18] +q[58] +q[62] -q[67] -q[68] +q[94] -q[103] -q[104] +q[146] -q[161] -q[162]);
  b.omega(i,j,k,22) = th.MW[21] * ( -q[19] +q[21] -q[69] -q[105] +q[108] +q[122] -q[170] -q[171]);
  b.omega(i,j,k,23) = th.MW[22] * ( -q[20] -q[21] -q[22] +q[69] -q[70] +q[72] -q[106] -q[107] -q[108] -q[109] +q[110] +q[123] +q[127] +q[133] +q[136] +q[171] +q[173] +q[176] +q[291] +q[294]);
  b.omega(i,j,k,24) = th.MW[23] * ( -q[23] +q[70] -q[71] -q[72] +q[74] -q[110] +q[111] +q[128] +q[140] +q[163] -q[172] -q[293] -q[294]);
  b.omega(i,j,k,25) = th.MW[24] * ( -q[24] +q[71] -q[73] -q[74] +q[76] -q[111] +q[129] +q[137] +q[148] -q[163] -q[173] +q[174] -q[284] -q[317]);
  b.omega(i,j,k,26) = th.MW[25] * ( -q[25] +q[26] +q[73] -q[75] -q[76] +q[77] +q[112] +q[153] +q[158] +q[164] -q[174] -q[285] -q[311] +q[318] +q[320] +q[321] +q[323] +2.0*q[324]);
  b.omega(i,j,k,27) = th.MW[26] * ( -q[26] +q[75] -q[77] -q[112] -q[153] +q[157] -q[164]);
  b.omega(i,j,k,28) = th.MW[27] * ( q[20] -q[27] +q[28] -q[78] +q[79] +q[105] +q[113] +q[130] -q[133] -q[140] -q[175] -2.0*q[176] -q[273]);
  b.omega(i,j,k,29) = th.MW[28] * ( q[23] -q[28] -q[29] -q[79] -q[80] +q[81] +q[106] -q[113] +q[132] +q[139] -q[303] +q[308] +q[309]);
  b.omega(i,j,k,30) = th.MW[29] * ( -q[81] +q[107]);
  b.omega(i,j,k,31) = th.MW[30] * ( -q[177] -q[178] -q[179] +q[190] +q[192] -q[195] +q[216] -q[224] +q[226] -q[237] +q[238] +q[239] +q[244] +q[247] -q[274] -q[275] -q[282]);
  b.omega(i,j,k,32) = th.MW[31] * ( -q[189] -q[190] -q[191] -q[192] -q[193] -q[194] -q[195] -q[196] -q[197] -q[198] +q[199] +q[201] +q[202] +q[207] +q[222] +q[231] +q[241] +q[242] +q[261] +q[268] -q[279]);
  b.omega(i,j,k,33) = th.MW[32] * ( -q[199] -q[200] -q[201] -q[202] +q[235] +q[264] +q[267] +q[271] +q[276] +q[277] +q[278]);
  b.omega(i,j,k,34) = th.MW[33] * ( -q[276] -q[277] -q[278]);
  b.omega(i,j,k,35) = th.MW[34] * ( -q[203] -q[204] -q[205] -q[206] -q[207] -q[208] -q[209] -q[210]);
  b.omega(i,j,k,36) = th.MW[35] * ( -q[177] +q[178] +q[179] +2.0*q[181] -q[185] -q[186] +q[187] +q[188] +q[189] +q[194] -q[197] -q[198] +q[207] -q[211] +q[212] +q[213] +q[214] +q[215] +q[221] +q[223] +q[225] -q[227] -q[228] -q[243] -q[244] -q[245] -q[246] -q[247] -q[248] -q[249] -q[250] -q[251] -q[252] -q[253] -q[254] -q[255] +q[257] -q[273] +q[280] +q[282]);
  b.omega(i,j,k,37) = th.MW[36] * ( q[185] +q[186] -q[187] -q[188] -q[280] -q[281]);
  b.omega(i,j,k,38) = th.MW[37] * ( -q[180] -q[181] -q[182] -q[183] -q[184] +q[198] +q[227] +q[281]);
  b.omega(i,j,k,39) = th.MW[38] * ( q[191] +q[193] +q[196] +q[200] +q[211] -q[212] -q[213] -q[214] -q[215] +q[262] +q[279]);
  b.omega(i,j,k,40) = th.MW[39] * ( -q[216] -q[217] -q[218] -q[219] -q[220] +q[229] +q[232] +q[238] +q[243] -q[280]);
  b.omega(i,j,k,41) = th.MW[40] * ( q[218] +q[220] -q[229] -q[230] -q[231] -q[232] -q[233] -q[234] -q[235] -q[236] +q[239] +q[241] +q[242] +q[245] +q[249] +q[252] +q[254] +q[257] +q[270] +q[275]);
  b.omega(i,j,k,42) = th.MW[41] * ( q[236] -q[237] +q[255] +q[274]);
  b.omega(i,j,k,43) = th.MW[42] * ( q[240] -q[256] -q[257] -q[258] -q[259] -q[260]);
  b.omega(i,j,k,44) = th.MW[43] * ( q[250] +q[253] -q[269] -q[270] -q[271] +q[273]);
  b.omega(i,j,k,45) = th.MW[44] * ( q[233] -q[272]);
  b.omega(i,j,k,46) = th.MW[45] * ( q[234] +q[248] +q[251] -q[261] -q[262] -q[263] -q[264] -q[265] -q[266] -q[267] -q[268] +q[269] +q[272]);
  b.omega(i,j,k,47) = th.MW[46] * ( q[217] +q[219] -q[221] -q[222] -q[223] -q[224] -q[225] -q[226] -q[227] -q[228] +q[230] +q[246] +q[263] +q[265] +q[266] +q[280] -q[281]);
  b.omega(i,j,k,48) = th.MW[47] * ( q[177] +q[180] +q[182] +q[183] +q[184] +q[195] +q[197] +q[203] +q[204] +q[205] +q[206] +q[208] +q[209] +q[210] +q[224] +q[228] +q[237] -q[238] -q[239] -q[240] -q[241] -q[242] +q[256] +q[258] +q[259] +q[260]);
  b.omega(i,j,k,49) = th.MW[48] * (0.0);
  b.omega(i,j,k,50) = th.MW[49] * ( q[312] +q[313] +q[314] -q[315] +q[316] +q[317] -q[318] -q[319] -q[320] -q[321] -q[322] -q[323] -q[324]);
  b.omega(i,j,k,51) = th.MW[50] * ( q[311] -q[312] -q[313] -q[314] +q[315] -q[316] +q[319] +q[322]);
  b.omega(i,j,k,52) = th.MW[51] * ( q[284] +q[293] +q[295] +q[298] +q[303] -q[304] -q[305] -q[306] -q[307] -q[308] -q[309] -q[310]);
  b.omega(i,j,k,53) = th.MW[52] * ( q[285] -q[295] -q[296] -q[297] -q[298] -q[299] -q[300] -q[301] -q[302]);

  // Add source terms to RHS
  for (int n=0; n<th.ns-1; n++)
  {
    b.dQ(i,j,k,5+n) += b.omega(i,j,k,n+1);
  }
  // Compute constant pressure dTdt dYdt (for implicit chem integration)
  double dTdt = 0.0;
  for (int n=0; n<=th.ns-1; n++)
  {
    dTdt -= b.qh(i,j,k,5+n) * b.omega(i,j,k,n+1);
    b.omega(i,j,k,n+1) /= b.Q(i,j,k,0);
  }
  dTdt /= b.qh(i,j,k,1) * b.Q(i,j,k,0);
  b.omega(i,j,k,0) = dTdt;

  token.release(id);
  });
}
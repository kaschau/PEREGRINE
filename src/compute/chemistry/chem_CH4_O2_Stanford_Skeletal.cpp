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
// ========================================================== //

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include "compute.hpp"
#include <math.h>
#include <vector>

void chem_CH4_O2_Stanford_Skeletal(std::vector<block_> mb, thtrdat_ th) {
for(block_ b : mb){

// --------------------------------------------------------------|
// cc range
// --------------------------------------------------------------|
  MDRange3 range = MDRange3({1,1,1},{b.ni,b.nj,b.nk});

  Kokkos::parallel_for("Compute chemical source terms",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  const int ns=12;
  const int nr=38;
  const int l_tbc=7;
  double T,logT,prefRuT;
  double Y[ns],cs[ns];

  double rho;

  T = b.q(i,j,k,4);
  logT = log(T);
  prefRuT = 101325.0/(th.Ru*T);
  rho = b.Q(i,j,k,0);

  // Compute nth species Y
  Y[ns-1] = 1.0;
  for (int n=0; n<ns-1; n++)
  {
    Y[n] = b.q(i,j,k,5+n);
    Y[ns-1] -= Y[n];
  }
  Y[ns-1] = std::max(0.0,Y[ns-1]);

  // Conecntrations
  for (int n=0; n<=ns-1; n++)
  {
    cs[n] = rho*Y[n]/th.MW[n];
  }

  // ----------------------------------------------------------- >
  // Chaperon efficiencies. ------------------------------------ >
  // ----------------------------------------------------------- >

  std::array<double, nr> S_tbc;
  S_tbc.fill(1.0);

  S_tbc[5] = 2.5*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + 12.0*cs[6] + cs[7] + 2.0*cs[8] + 1.9*cs[9] + 2.5*cs[10] + 3.8*cs[11];

  S_tbc[6] = 2.5*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + 12.0*cs[6] + cs[7] + 2.0*cs[8] + 1.9*cs[9] + 2.5*cs[10] + 3.8*cs[11];

  S_tbc[7] = 3.0*cs[0] + cs[1] + 1.5*cs[2] + cs[3] + cs[4] + cs[5] + cs[7] + 7.0*cs[8] + 1.9*cs[9] + 2.5*cs[10] + 3.8*cs[11];

  S_tbc[9] = 2.0*cs[0] + cs[1] + 0.78*cs[2] + cs[3] + cs[4] + cs[5] + 14.0*cs[6] + cs[7] + 2.0*cs[8] + 1.9*cs[9] + 2.5*cs[10] + 3.8*cs[11];

  S_tbc[16] = 2.5*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + 12.0*cs[6] + cs[7] + 2.0*cs[8] + 1.9*cs[9] + 2.5*cs[10] + 3.8*cs[11];

  S_tbc[24] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + 6.0*cs[6] + cs[7] + 2.0*cs[8] + 1.5*cs[9] + 2.5*cs[10] + 2.0*cs[11];

  S_tbc[32] = 2.0*cs[0] + cs[1] + cs[2] + cs[3] + cs[4] + cs[5] + 6.0*cs[6] + cs[7] + 2.0*cs[8] + 1.5*cs[9] + 2.5*cs[10] + 2.0*cs[11];

  // ----------------------------------------------------------- >
  // Gibbs energy. --------------------------------------------- >
  // ----------------------------------------------------------- >

  int m;
  double hi,scs;
  double gbs[ns];

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

    gbs[n] = hi-scs                         ;
  }

  // ----------------------------------------------------------- >
  // Rate Constants. ------------------------------------------- >
  // ----------------------------------------------------------- >

  double q_f[nr],k_f[nr];
  double q_b[nr],k_b[nr];

  double dG[nr],K_c[nr],q[nr]; 

  k_f[0] = exp(log(109000000000.00002)-(7704.291057866103/T));
   dG[0] =  - gbs[1] - gbs[2] + gbs[3] + gbs[4];
  K_c[0] = exp(-dG[0]);

  k_f[1] = exp(log(3820000000.0000005)-(4000.5952913151878/T));
   dG[1] =  - gbs[0] + gbs[1] - gbs[3] + gbs[4];
  K_c[1] = exp(-dG[1]);

  k_f[2] = exp(log(879000000000.0001)-(9651.750652506327/T));
   dG[2] =  - gbs[0] + gbs[1] - gbs[3] + gbs[4];
  K_c[2] = exp(-dG[2]);

  k_f[3] = exp(log(216000.00000000003)+1.51*logT-(1729.5655366352578/T));
   dG[3] =  - gbs[0] + gbs[1] - gbs[4] + gbs[6];
  K_c[3] = exp(-dG[3]);

  k_f[4] = exp(log(33.50000000000001)+2.42*logT-(-970.2072605856204/T));
   dG[4] =   gbs[3] -2.0*gbs[4] + gbs[6];
  K_c[4] = exp(-dG[4]);

  k_f[5] = exp(log(4.580000000000001e+16)-1.4*logT-(52531.08710193616/T));
   dG[5] =  - gbs[0] +2.0*gbs[1];
  K_c[5] = prefRuT*exp(-dG[5]);

  k_f[6] = exp(log(4710000000000.001)-1.0*logT);
   dG[6] =  - gbs[1] - gbs[3] + gbs[4];
  K_c[6] = exp(-dG[6])/prefRuT;

  k_f[7] = exp(log(6.060000000000001e+24)-3.322*logT-(60788.9196466509/T));
   dG[7] =   gbs[1] + gbs[4] - gbs[6];
  K_c[7] = prefRuT*exp(-dG[7]);

  k_f[8] = exp(log(1.0100000000000001e+23)-2.44*logT-(60486.98792655164/T));
   dG[8] =   gbs[1] + gbs[4] - gbs[6];
  K_c[8] = prefRuT*exp(-dG[8]);

  k_f[9] = exp(log(4650000000.000001)+0.44*logT);
   dG[9] =  - gbs[1] - gbs[2] + gbs[5];
  K_c[9] = exp(-dG[9])/prefRuT;

  k_f[10] = exp(log(3680.0000000000005)+2.087*logT-(-732.1844212407042/T));
   dG[10] =   gbs[0] - gbs[1] + gbs[2] - gbs[5];
  K_c[10] = exp(-dG[10]);

  k_f[11] = exp(log(70800000000.00002)-(150.96586004962973/T));
   dG[11] =  - gbs[1] +2.0*gbs[4] - gbs[5];
  K_c[11] = exp(-dG[11]);

  k_f[12] = 1450000000.0000002;
   dG[12] =  - gbs[1] + gbs[3] - gbs[5] + gbs[6];
  K_c[12] = exp(-dG[12]);

  k_f[13] = exp(log(16300000000.000002)-(-223.93269240695076/T));
   dG[13] =   gbs[2] - gbs[3] + gbs[4] - gbs[5];
  K_c[13] = exp(-dG[13]);

  k_f[14] = exp(log(7000000000.000001)-(-550.0189501141509/T));
   dG[14] =   gbs[2] - gbs[4] - gbs[5] + gbs[6];
  K_c[14] = exp(-dG[14]);

  k_f[15] = exp(log(450000000000.00006)-(5500.18950114151/T));
   dG[15] =   gbs[2] - gbs[4] - gbs[5] + gbs[6];
  K_c[15] = exp(-dG[15]);

  k_f[16] = exp(log(10600000000.000002)-0.308*logT-(3493.8532210819303/T));
   dG[16] =  - gbs[3] - gbs[9] + gbs[11];
  K_c[16] = exp(-dG[16])/prefRuT;

  k_f[17] = exp(log(2530000000.0000005)-(24003.571747891125/T));
   dG[17] =  - gbs[2] + gbs[3] - gbs[9] + gbs[11];
  K_c[17] = exp(-dG[17]);

  k_f[18] = exp(log(84.60000000000001)+2.053*logT-(-179.1461539255606/T));
   dG[18] =   gbs[1] - gbs[4] - gbs[9] + gbs[11];
  K_c[18] = exp(-dG[18]);

  k_f[19] = exp(log(8640000000.000002)-0.664*logT-(167.06888512159023/T));
   dG[19] =   gbs[1] - gbs[4] - gbs[9] + gbs[11];
  K_c[19] = exp(-dG[19]);

  k_f[20] = exp(log(157.00000000000003)+2.18*logT-(9029.771309101852/T));
   dG[20] =   gbs[4] - gbs[5] - gbs[9] + gbs[11];
  K_c[20] = exp(-dG[20]);

  k_f[21] = exp(log(3070.000000000001)+2.5*logT-(3818.4298201886345/T));
   dG[21] =   gbs[0] - gbs[1] + gbs[7] - gbs[8];
  K_c[21] = exp(-dG[21]);

  k_f[22] = exp(log(231000.00000000003)+1.56*logT-(4269.817741737027/T));
   dG[22] =  - gbs[3] + gbs[4] + gbs[7] - gbs[8];
  K_c[22] = exp(-dG[22]);

  k_f[23] = exp(log(1000.0000000000002)+2.182*logT-(1230.874978937981/T));
   dG[23] =  - gbs[4] + gbs[6] + gbs[7] - gbs[8];
  K_c[23] = exp(-dG[23]);

  k_f[24] = 141000000000.00003;
   dG[24] =  - gbs[1] - gbs[7] + gbs[8];
  K_c[24] = exp(-dG[24])/prefRuT;

  k_f[25] = 108000000000.00002;
   dG[25] =   gbs[1] - gbs[3] - gbs[7] + gbs[10];
  K_c[25] = exp(-dG[25]);

  k_f[26] = 23100000000.000004;
   dG[26] =   gbs[0] + gbs[1] - gbs[3] - gbs[7] + gbs[9];
  K_c[26] = prefRuT*exp(-dG[26]);

  k_f[27] = exp(log(116.00000000000001)+2.35*logT-(-765.9001299851215/T));
   dG[27] =   gbs[2] - gbs[5] - gbs[7] + gbs[8];
  K_c[27] = exp(-dG[27]);

  k_f[28] = exp(log(20800000000.000004)-(-296.8995247642718/T));
   dG[28] =   gbs[1] + gbs[4] - gbs[5] - gbs[7] + gbs[10];
  K_c[28] = prefRuT*exp(-dG[28]);

  k_f[29] = exp(log(2510000000.0000005)-(14239.603139414574/T));
   dG[29] =   gbs[1] - gbs[2] + gbs[3] - gbs[7] + gbs[10];
  K_c[29] = prefRuT*exp(-dG[29]);

  k_f[30] = exp(log(0.022800000000000004)+2.53*logT-(4915.4484032159435/T));
   dG[30] =  - gbs[2] + gbs[4] - gbs[7] + gbs[10];
  K_c[30] = exp(-dG[30]);

  k_f[31] = exp(log(0.010600000000000002)+3.36*logT-(2168.8761893796805/T));
   dG[31] =   gbs[1] - gbs[7] + gbs[8] + gbs[9] - gbs[10];
  K_c[31] = prefRuT*exp(-dG[31]);

  k_f[32] = exp(log(37000000000000.0)-(36219.729143107164/T));
   dG[32] =   gbs[0] + gbs[9] - gbs[10];
  K_c[32] = prefRuT*exp(-dG[32]);

  k_f[33] = exp(log(5670000000.000001)+0.361*logT-(2319.338829895811/T));
   dG[33] =   gbs[0] + gbs[9] - gbs[10];
  K_c[33] = prefRuT*exp(-dG[33]);

  k_f[34] = exp(log(11400000000.000002)+0.582*logT-(7243.845184714733/T));
   dG[34] =   gbs[0] + gbs[9] - gbs[10];
  K_c[34] = prefRuT*exp(-dG[34]);

  k_f[35] = exp(log(416000000.00000006)+0.57*logT-(1389.892351523591/T));
   dG[35] =   gbs[1] - gbs[3] + gbs[4] + gbs[9] - gbs[10];
  K_c[35] = prefRuT*exp(-dG[35]);

  k_f[36] = exp(log(78200.00000000001)+1.63*logT-(-530.8966078411978/T));
   dG[36] =   gbs[1] - gbs[4] + gbs[6] + gbs[9] - gbs[10];
  K_c[36] = prefRuT*exp(-dG[36]);

  k_f[37] = exp(log(244.00000000000006)+2.5*logT-(18347.384191365/T));
   dG[37] =   gbs[1] - gbs[2] + gbs[5] + gbs[9] - gbs[10];
  K_c[37] = prefRuT*exp(-dG[37]);

  // ----------------------------------------------------------- >
  // FallOff Modifications. ------------------------------------ >
  // ----------------------------------------------------------- >

  double Fcent[7];
  double pmod[7];
  double Pr,k0;
  double A,f1,F_pdr;
  double C,N;

  //  Three Body Reaction #5
  //  Three Body Reaction #6
  //  Three Body Reaction #7
  //  Troe Reaction #9
  Fcent[3] = (1.0 - (0.5))*exp(-T/(30.0)) + (0.5) *exp(-T/(90000.0)) + exp(-(90000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[3]);
  N =   0.75 - 1.27*log10(Fcent[3]);
  k0 = exp(log(1910000000000000.2)-1.72*logT-(264.190255086852/T));
  Pr = S_tbc[9]*k0/k_f[9];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[3])/(1.0+f1*f1));

  pmod[3] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[9] = k_f[9]*pmod[3];

  //  Lindeman Reaction #16
  Fcent[4] = 1.0;
  k0 = exp(log(1400000000000000.2)-2.1*logT-(2767.7074342432115/T));
  Pr = S_tbc[16]*k0/k_f[16];
  pmod[4] = Pr/(1.0 + Pr);
  k_f[16] = k_f[16]*pmod[4];

  //  Troe Reaction #24
  Fcent[5] = (1.0 - (0.37))*exp(-T/(3315.0)) + (0.37) *exp(-T/(61.0)) + exp(-(90000.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[5]);
  N =   0.75 - 1.27*log10(Fcent[5]);
  k0 = exp(log(6.35e+29)-5.57*logT-(1921.2921788982876/T));
  Pr = S_tbc[24]*k0/k_f[24];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[5])/(1.0+f1*f1));

  pmod[5] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[24] = k_f[24]*pmod[5];

  //  Troe Reaction #32
  Fcent[6] = (1.0 - (0.932))*exp(-T/(197.00000000000003)) + (0.932) *exp(-T/(1540.0)) + exp(-(10300.0)/T);
  C = - 0.4 - 0.67*log10(Fcent[6]);
  N =   0.75 - 1.27*log10(Fcent[6]);
  k0 = exp(log(4.4000000000000005e+35)-6.1*logT-(47302.636148883976/T));
  Pr = S_tbc[32]*k0/k_f[32];
  A = log10(Pr) + C;
  f1 = A/(N - 0.14*A);
  F_pdr = pow(10.0,log10(Fcent[6])/(1.0+f1*f1));

  pmod[6] =  Pr/(1.0 + Pr) * F_pdr;
  k_f[32] = k_f[32]*pmod[6];



  // ----------------------------------------------------------- >
  // Forward, backward, net rates of progress. ----------------- >
  // ----------------------------------------------------------- >

  q_f[0] =   S_tbc[0] * k_f[0] * cs[1] * cs[2];
  q_b[0] = - S_tbc[0] * k_f[0]/K_c[0] * cs[3] * cs[4];
  q[  0] =   q_f[0] + q_b[0];

  q_f[1] =   S_tbc[1] * k_f[1] * cs[0] * cs[3];
  q_b[1] = - S_tbc[1] * k_f[1]/K_c[1] * cs[1] * cs[4];
  q[  1] =   q_f[1] + q_b[1];

  q_f[2] =   S_tbc[2] * k_f[2] * cs[0] * cs[3];
  q_b[2] = - S_tbc[2] * k_f[2]/K_c[2] * cs[1] * cs[4];
  q[  2] =   q_f[2] + q_b[2];

  q_f[3] =   S_tbc[3] * k_f[3] * cs[0] * cs[4];
  q_b[3] = - S_tbc[3] * k_f[3]/K_c[3] * cs[1] * cs[6];
  q[  3] =   q_f[3] + q_b[3];

  q_f[4] =   S_tbc[4] * k_f[4] * pow(cs[4],2.0);
  q_b[4] = - S_tbc[4] * k_f[4]/K_c[4] * cs[3] * cs[6];
  q[  4] =   q_f[4] + q_b[4];

  q_f[5] =   S_tbc[5] * k_f[5] * cs[0];
  q_b[5] = - S_tbc[5] * k_f[5]/K_c[5] * pow(cs[1],2.0);
  q[  5] =   q_f[5] + q_b[5];

  q_f[6] =   S_tbc[6] * k_f[6] * cs[1] * cs[3];
  q_b[6] = - S_tbc[6] * k_f[6]/K_c[6] * cs[4];
  q[  6] =   q_f[6] + q_b[6];

  q_f[7] =   S_tbc[7] * k_f[7] * cs[6];
  q_b[7] = - S_tbc[7] * k_f[7]/K_c[7] * cs[1] * cs[4];
  q[  7] =   q_f[7] + q_b[7];

  q_f[8] =   S_tbc[8] * k_f[8] * pow(cs[6],2.0);
  q_b[8] = - S_tbc[8] * k_f[8]/K_c[8] * cs[1] * cs[4] * cs[6];
  q[  8] =   q_f[8] + q_b[8];

  q_f[9] =   k_f[9] * cs[1] * cs[2];
  q_b[9] = - k_f[9]/K_c[9] * cs[5];
  q[  9] =   q_f[9] + q_b[9];

  q_f[10] =   S_tbc[10] * k_f[10] * cs[1] * cs[5];
  q_b[10] = - S_tbc[10] * k_f[10]/K_c[10] * cs[0] * cs[2];
  q[  10] =   q_f[10] + q_b[10];

  q_f[11] =   S_tbc[11] * k_f[11] * cs[1] * cs[5];
  q_b[11] = - S_tbc[11] * k_f[11]/K_c[11] * pow(cs[4],2.0);
  q[  11] =   q_f[11] + q_b[11];

  q_f[12] =   S_tbc[12] * k_f[12] * cs[1] * cs[5];
  q_b[12] = - S_tbc[12] * k_f[12]/K_c[12] * cs[3] * cs[6];
  q[  12] =   q_f[12] + q_b[12];

  q_f[13] =   S_tbc[13] * k_f[13] * cs[3] * cs[5];
  q_b[13] = - S_tbc[13] * k_f[13]/K_c[13] * cs[2] * cs[4];
  q[  13] =   q_f[13] + q_b[13];

  q_f[14] =   S_tbc[14] * k_f[14] * cs[4] * cs[5];
  q_b[14] = - S_tbc[14] * k_f[14]/K_c[14] * cs[2] * cs[6];
  q[  14] =   q_f[14] + q_b[14];

  q_f[15] =   S_tbc[15] * k_f[15] * cs[4] * cs[5];
  q_b[15] = - S_tbc[15] * k_f[15]/K_c[15] * cs[2] * cs[6];
  q[  15] =   q_f[15] + q_b[15];

  q_f[16] =   k_f[16] * cs[3] * cs[9];
  q_b[16] = - k_f[16]/K_c[16] * cs[11];
  q[  16] =   q_f[16] + q_b[16];

  q_f[17] =   S_tbc[17] * k_f[17] * cs[2] * cs[9];
  q_b[17] = - S_tbc[17] * k_f[17]/K_c[17] * cs[3] * cs[11];
  q[  17] =   q_f[17] + q_b[17];

  q_f[18] =   S_tbc[18] * k_f[18] * cs[4] * cs[9];
  q_b[18] = - S_tbc[18] * k_f[18]/K_c[18] * cs[1] * cs[11];
  q[  18] =   q_f[18] + q_b[18];

  q_f[19] =   S_tbc[19] * k_f[19] * cs[4] * cs[9];
  q_b[19] = - S_tbc[19] * k_f[19]/K_c[19] * cs[1] * cs[11];
  q[  19] =   q_f[19] + q_b[19];

  q_f[20] =   S_tbc[20] * k_f[20] * cs[5] * cs[9];
  q_b[20] = - S_tbc[20] * k_f[20]/K_c[20] * cs[4] * cs[11];
  q[  20] =   q_f[20] + q_b[20];

  q_f[21] =   S_tbc[21] * k_f[21] * cs[1] * cs[8];
  q_b[21] = - S_tbc[21] * k_f[21]/K_c[21] * cs[0] * cs[7];
  q[  21] =   q_f[21] + q_b[21];

  q_f[22] =   S_tbc[22] * k_f[22] * cs[3] * cs[8];
  q_b[22] = - S_tbc[22] * k_f[22]/K_c[22] * cs[4] * cs[7];
  q[  22] =   q_f[22] + q_b[22];

  q_f[23] =   S_tbc[23] * k_f[23] * cs[4] * cs[8];
  q_b[23] = - S_tbc[23] * k_f[23]/K_c[23] * cs[6] * cs[7];
  q[  23] =   q_f[23] + q_b[23];

  q_f[24] =   k_f[24] * cs[1] * cs[7];
  q_b[24] = - k_f[24]/K_c[24] * cs[8];
  q[  24] =   q_f[24] + q_b[24];

  q_f[25] =   S_tbc[25] * k_f[25] * cs[3] * cs[7];
  q_b[25] = - S_tbc[25] * k_f[25]/K_c[25] * cs[1] * cs[10];
  q[  25] =   q_f[25] + q_b[25];

  q_f[26] =   S_tbc[26] * k_f[26] * cs[3] * cs[7];
  q_b[26] = - S_tbc[26] * k_f[26]/K_c[26] * cs[0] * cs[1] * cs[9];
  q[  26] =   q_f[26];

  q_f[27] =   S_tbc[27] * k_f[27] * cs[5] * cs[7];
  q_b[27] = - S_tbc[27] * k_f[27]/K_c[27] * cs[2] * cs[8];
  q[  27] =   q_f[27] + q_b[27];

  q_f[28] =   S_tbc[28] * k_f[28] * cs[5] * cs[7];
  q_b[28] = - S_tbc[28] * k_f[28]/K_c[28] * cs[1] * cs[4] * cs[10];
  q[  28] =   q_f[28];

  q_f[29] =   S_tbc[29] * k_f[29] * cs[2] * cs[7];
  q_b[29] = - S_tbc[29] * k_f[29]/K_c[29] * cs[1] * cs[3] * cs[10];
  q[  29] =   q_f[29];

  q_f[30] =   S_tbc[30] * k_f[30] * cs[2] * cs[7];
  q_b[30] = - S_tbc[30] * k_f[30]/K_c[30] * cs[4] * cs[10];
  q[  30] =   q_f[30] + q_b[30];

  q_f[31] =   S_tbc[31] * k_f[31] * cs[7] * cs[10];
  q_b[31] = - S_tbc[31] * k_f[31]/K_c[31] * cs[1] * cs[8] * cs[9];
  q[  31] =   q_f[31];

  q_f[32] =   k_f[32] * cs[10];
  q_b[32] = - k_f[32]/K_c[32] * cs[0] * cs[9];
  q[  32] =   q_f[32] + q_b[32];

  q_f[33] =   S_tbc[33] * k_f[33] * cs[1] * cs[10];
  q_b[33] = - S_tbc[33] * k_f[33]/K_c[33] * cs[0] * cs[1] * cs[9];
  q[  33] =   q_f[33];

  q_f[34] =   S_tbc[34] * k_f[34] * cs[1] * cs[10];
  q_b[34] = - S_tbc[34] * k_f[34]/K_c[34] * cs[0] * cs[1] * cs[9];
  q[  34] =   q_f[34];

  q_f[35] =   S_tbc[35] * k_f[35] * cs[3] * cs[10];
  q_b[35] = - S_tbc[35] * k_f[35]/K_c[35] * cs[1] * cs[4] * cs[9];
  q[  35] =   q_f[35];

  q_f[36] =   S_tbc[36] * k_f[36] * cs[4] * cs[10];
  q_b[36] = - S_tbc[36] * k_f[36]/K_c[36] * cs[1] * cs[6] * cs[9];
  q[  36] =   q_f[36];

  q_f[37] =   S_tbc[37] * k_f[37] * cs[2] * cs[10];
  q_b[37] = - S_tbc[37] * k_f[37]/K_c[37] * cs[1] * cs[5] * cs[9];
  q[  37] =   q_f[37];

  // ----------------------------------------------------------- >
  // Source terms. --------------------------------------------- >
  // ----------------------------------------------------------- >

  b.omega(i,j,k,0) = th.MW[0] * ( -q[1] -q[2] -q[3] -q[5] +q[10] +q[21] +q[26] +q[32] +q[33] +q[34]);
  b.omega(i,j,k,1) = th.MW[1] * ( -q[0] +q[1] +q[2] +q[3] +2.0*q[5] -q[6] +q[7] +q[8] -q[9] -q[10] -q[11] -q[12] +q[18] +q[19] -q[21] -q[24] +q[25] +q[26] +q[28] +q[29] +q[31] +q[35] +q[36] +q[37]);
  b.omega(i,j,k,2) = th.MW[2] * ( -q[0] -q[9] +q[10] +q[13] +q[14] +q[15] -q[17] +q[27] -q[29] -q[30] -q[37]);
  b.omega(i,j,k,3) = th.MW[3] * ( q[0] -q[1] -q[2] +q[4] -q[6] +q[12] -q[13] -q[16] +q[17] -q[22] -q[25] -q[26] +q[29] -q[35]);
  b.omega(i,j,k,4) = th.MW[4] * ( q[0] +q[1] +q[2] -q[3] -2.0*q[4] +q[6] +q[7] +q[8] +2.0*q[11] +q[13] -q[14] -q[15] -q[18] -q[19] +q[20] +q[22] -q[23] +q[28] +q[30] +q[35] -q[36]);
  b.omega(i,j,k,5) = th.MW[5] * ( q[9] -q[10] -q[11] -q[12] -q[13] -q[14] -q[15] -q[20] -q[27] -q[28] +q[37]);
  b.omega(i,j,k,6) = th.MW[6] * ( q[3] +q[4] -q[7] -q[8] +q[12] +q[14] +q[15] +q[23] +q[36]);
  b.omega(i,j,k,7) = th.MW[7] * ( q[21] +q[22] +q[23] -q[24] -q[25] -q[26] -q[27] -q[28] -q[29] -q[30] -q[31]);
  b.omega(i,j,k,8) = th.MW[8] * ( -q[21] -q[22] -q[23] +q[24] +q[27] +q[31]);
  b.omega(i,j,k,9) = th.MW[9] * ( -q[16] -q[17] -q[18] -q[19] -q[20] +q[26] +q[31] +q[32] +q[33] +q[34] +q[35] +q[36] +q[37]);
  b.omega(i,j,k,10) = th.MW[10] * ( q[25] +q[28] +q[29] +q[30] -q[31] -q[32] -q[33] -q[34] -q[35] -q[36] -q[37]);
  b.omega(i,j,k,11) = th.MW[11] * ( q[16] +q[17] +q[18] +q[19] +q[20]);

  // Add source terms to RHS
  for (int n=0; n<th.ns-1; n++)
  {
    b.dQ(i,j,k,5+n) += b.omega(i,j,k,n);
  }

  });
}}
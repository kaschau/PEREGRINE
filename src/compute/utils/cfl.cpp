#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "vector"
#include "array"
#include "math.h"

std::array<double, 2> CFLmax(std::vector<block_> mb) {

//-------------------------------------------------------------------------------------------|
// Compute the max acoustic and convective CFL factor speed/dx
//-------------------------------------------------------------------------------------------|
  double CFLmaxA, CFLmaxC;
  double returnMaxA=0.0;
  double returnMaxC=0.0;

  for (const block_ b: mb) {
    MDRange3 range_cc({b.ng, b.ng, b.ng},{b.ni+b.ng-1,
                                          b.nj+b.ng-1,
                                          b.nk+b.ng-1});
    Kokkos::parallel_reduce("CFL",
                            range_cc,
                            KOKKOS_LAMBDA(const int i,
                                          const int j,
                                          const int k,
                                          double& CFLA,
                                          double& CFLC) {

      // Cell lengths
      double dI = sqrt(pow(b.ixc(i+1,j  ,k  ) - b.ixc(i,j,k), 2.0) +
                       pow(b.iyc(i+1,j  ,k  ) - b.iyc(i,j,k), 2.0) +
                       pow(b.izc(i+1,j  ,k  ) - b.izc(i,j,k), 2.0) );
      double dJ = sqrt(pow(b.jxc(i  ,j+1,k  ) - b.jxc(i,j,k), 2.0) +
                       pow(b.jyc(i  ,j+1,k  ) - b.jyc(i,j,k), 2.0) +
                       pow(b.jzc(i  ,j+1,k  ) - b.jzc(i,j,k), 2.0) );
      double dK = sqrt(pow(b.kxc(i  ,j  ,k+1) - b.kxc(i,j,k), 2.0) +
                       pow(b.kyc(i  ,j  ,k+1) - b.kyc(i,j,k), 2.0) +
                       pow(b.kzc(i  ,j  ,k+1) - b.kzc(i,j,k), 2.0) );

      // Find max convective CFL
      double u = sqrt(pow(b.q(i,j,k,1), 2.0) +
                      pow(b.q(i,j,k,2), 2.0) +
                      pow(b.q(i,j,k,3), 2.0) );

      double uI = abs(0.5*(b.inx(i  ,j  ,k  )+b.inx(i+1,j  ,k  ))*u);
      double uJ = abs(0.5*(b.jny(i  ,j  ,k  )+b.jnx(i  ,j+1,k  ))*u);
      double uK = abs(0.5*(b.kny(i  ,j  ,k  )+b.knx(i  ,j  ,k+1))*u);

      double& c = b.qh(i,j,k,3);

      if (b.ni > 2) {
          CFLA = fmax(CFLA, c/dI);
          CFLC = fmax(CFLC, uI/dI);
      }
      if (b.nj > 2) {
          CFLA = fmax(CFLA, c/dJ);
          CFLC = fmax(CFLC, uJ/dJ);
      }
      if (b.nk > 2) {
          CFLA = fmax(CFLA, c/dK);
          CFLC = fmax(CFLC, uK/dK);
      }

    }, Kokkos::Max<double>(CFLmaxA),
       Kokkos::Max<double>(CFLmaxC));
  }

  returnMaxA = fmax(CFLmaxA, returnMaxA);
  returnMaxC = fmax(CFLmaxC, returnMaxC);

  return {returnMaxA, returnMaxC};

}

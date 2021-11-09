#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"

void fourthOrderKEEP(block_ b, const thtrdat_ th) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng,b.ng,b.ng},{b.ni+b.ng, b.nj+b.ng-1, b.nk+b.ng-1});

  const int order = 4;
  constexpr int q = order/2;
  constexpr int narray = (q*q+q)/2;
  constexpr double aq[2] = {2.0/3.0, -1.0/12.0};

  Kokkos::parallel_for("i face conv fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double U;
    double uf=0.0;
    double vf=0.0;
    double wf=0.0;

    // Reusable arrays
    double uR[narray],vR[narray],wR[narray],rhoR[narray];

    double a;
    // Compute face normal volume flux vector
    double tempu;
    double tempv;
    double tempw;
    int count=0;
    for (int is=1; is <= q; is++) {
      tempu = 0.0;
      tempv = 0.0;
      tempw = 0.0;
      for (int js=0; js <= is-1; js++) {
        uR[count] = 0.5*( b.q(i+js,j,k,1) + b.q(i+js-is,j,k,1) );
        vR[count] = 0.5*( b.q(i+js,j,k,2) + b.q(i+js-is,j,k,2) );
        wR[count] = 0.5*( b.q(i+js,j,k,3) + b.q(i+js-is,j,k,3) );
        tempu += uR[count];
        tempv += vR[count];
        tempw += wR[count];
        count++;
      }
      a = aq[is-1];
      uf += a*tempu;
      vf += a*tempv;
      wf += a*tempw;
    }
    uf *=2.0;
    vf *=2.0;
    wf *=2.0;

    U = b.isx(i,j,k)*uf +
        b.isy(i,j,k)*vf +
        b.isz(i,j,k)*wf ;

    //Compute fluxes

    // Continuity rho*Ui
    double rho = 0.0;
    double temprho = 0.0;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprho = 0.0;
      for (int js=0; js <= is-1; js++) {
        rhoR[count] = 0.5*( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) );
        temprho += rhoR[count];
        count++;
      }
      rho += a*temprho;
    }
    rho *=2.0;

    b.iF(i,j,k,0) = rho * U;

    // x momentum rho*u*Ui+ p*Ax
    // y momentum rho*v*Ui+ p*Ay
    // w momentum rho*w*Ui+ p*Az
    double rhou=0.0;
    double rhov=0.0;
    double rhow=0.0;
    double p=0.0;
    double temprhou, temprhov, temprhow, tempp;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhou = 0.0;
      temprhov = 0.0;
      temprhow = 0.0;
      tempp = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprhou += rhoR[count] * uR[count];
        temprhov += rhoR[count] * vR[count];
        temprhow += rhoR[count] * wR[count];
        tempp    += 0.5*( b.q(i+js,j,k,0) + b.q(i+js-is,j,k,0) );
        count++;
      }
      rhou += a*temprhou;
      rhov += a*temprhov;
      rhow += a*temprhow;
      p += a*tempp;
    }
    rhow *=2.0;
    rhov *=2.0;
    rhou *=2.0;
    p *=2.0;

    b.iF(i,j,k,1) = rhou * U + p*b.isx(i,j,k) ;

    b.iF(i,j,k,2) = rhov * U + p*b.isy(i,j,k) ;

    b.iF(i,j,k,3) = rhow * U + p*b.isz(i,j,k) ;

    // Total energy (rhoE+ p)*Ui)
    double rhoE=0.0;
    double pu=0.0;
    double temprhoE, temppu;
    double e,em;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhoE = 0.0;
      temppu = 0.0;
      for (int js=0; js <= is-1; js++) {
        e = b.qh(i+js   ,j,k,4)/b.Q(i+js   ,j,k,0);
        em= b.qh(i+js-is,j,k,4)/b.Q(i+js-is,j,k,0);

        temprhoE +=  rhoR[count] * ( 0.5*(  e + em )
                                   + 0.5*(  b.q(i+js,j,k,1)*b.q(i+js-is,j,k,1)
                                          + b.q(i+js,j,k,2)*b.q(i+js-is,j,k,2)
                                          + b.q(i+js,j,k,3)*b.q(i+js-is,j,k,3) )
                                   );
        count++;

        temppu += 0.5*(
                        b.q(i+js-is,j,k,0)*(  b.q(i+js   ,j,k,1)*b.isx(i,j,k)
                                            + b.q(i+js   ,j,k,2)*b.isy(i,j,k)
                                            + b.q(i+js   ,j,k,3)*b.isz(i,j,k) ) +
                        b.q(i+js   ,j,k,0)*(  b.q(i+js-is,j,k,1)*b.isx(i,j,k)
                                            + b.q(i+js-is,j,k,2)*b.isy(i,j,k)
                                            + b.q(i+js-is,j,k,3)*b.isz(i,j,k) )
                       );
      }
      rhoE += a*temprhoE;
      pu   += a*temppu;
    }
    rhoE *=2.0;
    pu   *=2.0;

    b.iF(i,j,k,4) = rhoE * U + pu;

    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      double rhoY = 0.0;
      double temprhoY = 0.0;
      count = 0;
      for (int is=1; is <= order/2; is++) {
        a = aq[is-1];
        temprhoY = 0.0;
        for (int js=0; js <= is-1; js++) {
          temprhoY += rhoR[count] * 0.5*(b.q(i+js,j,k,5+n)+b.q(i+js-is,j,k,5+n));
          count++;
        }
        rhoY += a*temprhoY;
      }
      rhoY *=2.0;
      b.iF(i,j,k,5+n) = rhoY * U;
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng, b.nk+b.ng-1});
  Kokkos::parallel_for("j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double V;
    double uf=0.0;
    double vf=0.0;
    double wf=0.0;

    // Reusable arrays
    double uR[narray],vR[narray],wR[narray],rhoR[narray];

    double a;
    // Compute face normal volume flux vector
    double tempu;
    double tempv;
    double tempw;
    int count=0;
    for (int is=1; is <= q; is++) {
      tempu = 0.0;
      tempv = 0.0;
      tempw = 0.0;
      for (int js=0; js <= is-1; js++) {
        uR[count] = 0.5*( b.q(i,j+js,k,1) + b.q(i,j+js-is,k,1) );
        vR[count] = 0.5*( b.q(i,j+js,k,2) + b.q(i,j+js-is,k,2) );
        wR[count] = 0.5*( b.q(i,j+js,k,3) + b.q(i,j+js-is,k,3) );
        tempu += uR[count];
        tempv += vR[count];
        tempw += wR[count];
        count++;
      }
      a = aq[is-1];
      uf += a*tempu;
      vf += a*tempv;
      wf += a*tempw;
    }
    uf *=2.0;
    vf *=2.0;
    wf *=2.0;

    V = b.jsx(i,j,k)*uf +
        b.jsy(i,j,k)*vf +
        b.jsz(i,j,k)*wf ;

    //Compute fluxes

    // Continuity rho*Vi
    double rho = 0.0;
    double temprho = 0.0;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprho = 0.0;
      for (int js=0; js <= is-1; js++) {
        rhoR[count] = 0.5*( b.Q(i,j+js,k,0) + b.Q(i,j+js-is,k,0) );
        temprho += rhoR[count];
        count++;
      }
      rho += a*temprho;
    }
    rho *=2.0;

    b.jF(i,j,k,0) = rho * V;

    // x momentum rho*u*Vi+ p*Ax
    // y momentum rho*v*Vi+ p*Ay
    // w momentum rho*w*Vi+ p*Az
    double rhou=0.0;
    double rhov=0.0;
    double rhow=0.0;
    double p=0.0;
    double temprhou, temprhov, temprhow, tempp;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhou = 0.0;
      temprhov = 0.0;
      temprhow = 0.0;
      tempp = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprhou += rhoR[count] * uR[count];
        temprhov += rhoR[count] * vR[count];
        temprhow += rhoR[count] * wR[count];
        tempp    += 0.5*( b.q(i,j+js,k,0) + b.q(i,j+js-is,k,0) );
        count++;
      }
      rhou += a*temprhou;
      rhov += a*temprhov;
      rhow += a*temprhow;
      p += a*tempp;
    }
    rhow *=2.0;
    rhov *=2.0;
    rhou *=2.0;
    p *=2.0;

    b.jF(i,j,k,1) = rhou * V + p*b.jsx(i,j,k) ;

    b.jF(i,j,k,2) = rhov * V + p*b.jsy(i,j,k) ;

    b.jF(i,j,k,3) = rhow * V + p*b.jsz(i,j,k) ;

    // Total energy (rhoE+ p)*Vi)
    double rhoE=0.0;
    double pu=0.0;
    double temprhoE, temppu;
    double e,em;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhoE = 0.0;
      temppu = 0.0;
      for (int js=0; js <= is-1; js++) {
        e = b.qh(i,j+js   ,k,4)/b.Q(i,j+js   ,k,0);
        em= b.qh(i,j+js-is,k,4)/b.Q(i,j+js-is,k,0);

        temprhoE +=  rhoR[count] * ( 0.5*(  e + em )
                                   + 0.5*(  b.q(i,j+js,k,1)*b.q(i,j+js-is,k,1)
                                          + b.q(i,j+js,k,2)*b.q(i,j+js-is,k,2)
                                          + b.q(i,j+js,k,3)*b.q(i,j+js-is,k,3) )
                                   );
        count++;

        temppu += 0.5*(
                        b.q(i,j+js-is,k,0)*(  b.q(i,j+js   ,k,1)*b.jsx(i,j,k)
                                            + b.q(i,j+js   ,k,2)*b.jsy(i,j,k)
                                            + b.q(i,j+js   ,k,3)*b.jsz(i,j,k) ) +
                        b.q(i,j+js   ,k,0)*(  b.q(i,j+js-is,k,1)*b.jsx(i,j,k)
                                            + b.q(i,j+js-is,k,2)*b.jsy(i,j,k)
                                            + b.q(i,j+js-is,k,3)*b.jsz(i,j,k) )
                       );
      }
      rhoE += a*temprhoE;
      pu   += a*temppu;
    }
    rhoE *=2.0;
    pu   *=2.0;

    b.jF(i,j,k,4) = rhoE * V + pu;

    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      double rhoY = 0.0;
      double temprhoY = 0.0;
      count = 0;
      for (int is=1; is <= order/2; is++) {
        a = aq[is-1];
        temprhoY = 0.0;
        for (int js=0; js <= is-1; js++) {
          temprhoY += rhoR[count] * 0.5*(b.q(i,j+js,k,5+n)+b.q(i,j+js-is,k,5+n));
          count++;
        }
        rhoY += a*temprhoY;
      }
      rhoY *=2.0;
      b.jF(i,j,k,5+n) = rhoY * V;
    }

  });

//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng-1, b.nk+b.ng});
  Kokkos::parallel_for("k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double W;
    double uf=0.0;
    double vf=0.0;
    double wf=0.0;

    // Reusable arrays
    double uR[narray],vR[narray],wR[narray],rhoR[narray];

    double a;
    // Compute face normal volume flux vector
    double tempu;
    double tempv;
    double tempw;
    int count=0;
    for (int is=1; is <= q; is++) {
      tempu = 0.0;
      tempv = 0.0;
      tempw = 0.0;
      for (int js=0; js <= is-1; js++) {
        uR[count] = 0.5*( b.q(i,j,k+js,1) + b.q(i,j,k+js-is,1) );
        vR[count] = 0.5*( b.q(i,j,k+js,2) + b.q(i,j,k+js-is,2) );
        wR[count] = 0.5*( b.q(i,j,k+js,3) + b.q(i,j,k+js-is,3) );
        tempu += uR[count];
        tempv += vR[count];
        tempw += wR[count];
        count++;
      }
      a = aq[is-1];
      uf += a*tempu;
      vf += a*tempv;
      wf += a*tempw;
    }
    uf *=2.0;
    vf *=2.0;
    wf *=2.0;

    W = b.ksx(i,j,k)*uf +
        b.ksy(i,j,k)*vf +
        b.ksz(i,j,k)*wf ;

    //Compute fluxes

    // Continuity rho*Wi
    double rho = 0.0;
    double temprho = 0.0;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprho = 0.0;
      for (int js=0; js <= is-1; js++) {
        rhoR[count] = 0.5*( b.Q(i,j,k+js,0) + b.Q(i,j,k+js-is,0) );
        temprho += rhoR[count];
        count++;
      }
      rho += a*temprho;
    }
    rho *=2.0;

    b.kF(i,j,k,0) = rho * W;

    // x momentum rho*u*Wi+ p*Ax
    // y momentum rho*v*Wi+ p*Ay
    // w momentum rho*w*Wi+ p*Az
    double rhou=0.0;
    double rhov=0.0;
    double rhow=0.0;
    double p=0.0;
    double temprhou, temprhov, temprhow, tempp;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhou = 0.0;
      temprhov = 0.0;
      temprhow = 0.0;
      tempp = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprhou += rhoR[count] * uR[count];
        temprhov += rhoR[count] * vR[count];
        temprhow += rhoR[count] * wR[count];
        tempp    += 0.5*( b.q(i,j,k+js,0) + b.q(i,j,k+js-is,0) );
        count++;
      }
      rhou += a*temprhou;
      rhov += a*temprhov;
      rhow += a*temprhow;
      p += a*tempp;
    }
    rhow *=2.0;
    rhov *=2.0;
    rhou *=2.0;
    p *=2.0;

    b.kF(i,j,k,1) = rhou * W + p*b.ksx(i,j,k) ;

    b.kF(i,j,k,2) = rhov * W + p*b.ksy(i,j,k) ;

    b.kF(i,j,k,3) = rhow * W + p*b.ksz(i,j,k) ;

    // Total energy (rhoE+ p)*Wi)
    double rhoE=0.0;
    double pu=0.0;
    double temprhoE, temppu;
    double e,em;
    count = 0;
    for (int is=1; is <= q; is++) {
      a = aq[is-1];
      temprhoE = 0.0;
      temppu = 0.0;
      for (int js=0; js <= is-1; js++) {
        e = b.qh(i,j,k+js   ,4)/b.Q(i,j,k+js   ,0);
        em= b.qh(i,j,k+js-is,4)/b.Q(i,j,k+js-is,0);

        temprhoE +=  rhoR[count] * ( 0.5*(  e + em )
                                   + 0.5*(  b.q(i,j,k+js,1)*b.q(i,j,k+js-is,1)
                                          + b.q(i,j,k+js,2)*b.q(i,j,k+js-is,2)
                                          + b.q(i,j,k+js,3)*b.q(i,j,k+js-is,3) )
                                   );
        count++;

        temppu += 0.5*(
                        b.q(i,j,k+js-is,0)*(  b.q(i,j,k+js   ,1)*b.ksx(i,j,k)
                                            + b.q(i,j,k+js   ,2)*b.ksy(i,j,k)
                                            + b.q(i,j,k+js   ,3)*b.ksz(i,j,k) ) +
                        b.q(i,j,k+js   ,0)*(  b.q(i,j,k+js-is,1)*b.ksx(i,j,k)
                                            + b.q(i,j,k+js-is,2)*b.ksy(i,j,k)
                                            + b.q(i,j,k+js-is,3)*b.ksz(i,j,k) )
                       );
      }
      rhoE += a*temprhoE;
      pu   += a*temppu;
    }
    rhoE *=2.0;
    pu   *=2.0;

    b.kF(i,j,k,4) = rhoE * W + pu;

    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      double rhoY = 0.0;
      double temprhoY = 0.0;
      count = 0;
      for (int is=1; is <= order/2; is++) {
        a = aq[is-1];
        temprhoY = 0.0;
        for (int js=0; js <= is-1; js++) {
          temprhoY += rhoR[count] * 0.5*(b.q(i,j,k+js,5+n)+b.q(i,j,k+js-is,5+n));
          count++;
        }
        rhoY += a*temprhoY;
      }
      rhoY *=2.0;
      b.kF(i,j,k,5+n) = rhoY * W;
    }

  });

}

#ifndef __block__H__
#define __block__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct block_ {
  int nblki;
  int ni,nj,nk;
  int ne=5;

  // Grid Arrays
  threeDview x,y,z;
  // Metric Arrays
  // Cell Centers
  threeDview xc,yc,zc;
  // i face area vectors
  threeDview isx,isy,isz,iS,inx,iny,inz;
  // j face area vectors
  threeDview jsx,jsy,jsz,jS,jnx,jny,jnz;
  // k face area vectors
  threeDview ksx,ksy,ksz,kS,knx,kny,knz;

  // Cons,Prim Arrays
  fourDview Q,q,dQ;

  // RHS stages
  fourDview rhs0,rhs1,rhs2,rhs3;

  // Flux Arrays
  fourDview iF,jF,kF;

};

#endif

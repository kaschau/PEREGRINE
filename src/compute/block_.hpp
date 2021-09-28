#ifndef __block__H__
#define __block__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct block_ {
  int nblki;
  int ni,nj,nk;
  int ne;
  int ns;

  // Grid Arrays
  threeDview x,y,z;
  // Metric Arrays
  // Cell Centers
  threeDview xc,yc,zc,J;
  threeDview dEdx,dEdy,dEdz;
  threeDview dNdx,dNdy,dNdz;
  threeDview dXdx,dXdy,dXdz;
  // i face area vectors
  threeDview isx,isy,isz,iS,inx,iny,inz;
  // j face area vectors
  threeDview jsx,jsy,jsz,jS,jnx,jny,jnz;
  // k face area vectors
  threeDview ksx,ksy,ksz,kS,knx,kny,knz;

  // Cons,Prim Arrays
  fourDview Q,q,dQ;
  // Spatial derivative of prim array
  fourDview dqdx,dqdy,dqdz;
  // thermo,trans arrays
  fourDview qh,qt;
  // chemistry
  fourDview omega;

  // RHS stages
  fourDview rhs0,rhs1,rhs2,rhs3;

  // Flux Arrays
  fourDview iF,jF,kF;

  // Flux Switch Array
  threeDview phi;

};

#endif

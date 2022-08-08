#ifndef __block__H__
#define __block__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data
// arrays for each block. Also converted into python class for modifying in the
// python wrapper
struct block_ {
  int nblki;
  int ni, nj, nk;
  int ng;
  int ne;
#ifdef NSCOMPILE
  const int ns = NS;
#endif

  // Grid Arrays
  threeDview x, y, z;
  // Metric Arrays
  // Cell Centers
  threeDview xc, yc, zc, J;
  threeDview dEdx, dEdy, dEdz;
  threeDview dNdx, dNdy, dNdz;
  threeDview dXdx, dXdy, dXdz;
  // i face centers
  threeDview ixc, iyc, izc;
  // i face area vectors
  threeDview isx, isy, isz, iS, inx, iny, inz;
  // j face centers
  threeDview jxc, jyc, jzc;
  // j face area vectors
  threeDview jsx, jsy, jsz, jS, jnx, jny, jnz;
  // k face centers
  threeDview kxc, kyc, kzc;
  // k face area vectors
  threeDview ksx, ksy, ksz, kS, knx, kny, knz;

  // Cons,Prim Arrays
  fourDview Q, q, dQ;
  // Spatial derivative of prim array
  fourDview dqdx, dqdy, dqdz;
  // thermo,trans arrays
  fourDview qh, qt;
  // chemistry
  fourDview omega;

  // Time integration stages
  fourDview Q0, Q1, Q2, Q3;
  fourDview Qn, Qnm1;
  threeDview dtau;

  // Flux Arrays
  fourDview iF, jF, kF;

  // Flux Switch Array
  fourDview phi;
};

#endif

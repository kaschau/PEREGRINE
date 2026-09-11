#ifndef __block__H__
#define __block__H__

#include "kokkosTypes.hpp"

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
  fourDview nodes;
  // Metric Arrays
  // Cell Centers
  fourDview cells;
  threeDview J;
  // distance between opposite face centers along each index direction
  fourDview dIJK;
  fiveDview dENCdxyz;

  // i face centers
  fourDview iFaces;
  // i face area vectors
  fourDview iS;
  // j face centers
  fourDview jFaces;
  // j face area vectors
  fourDview jS;
  // k face centers
  fourDview kFaces;
  // k face area vectors
  fourDview kS;

  // Cons,Prim Arrays
  fourDview Q, q, dQ;
  // Spatial derivative of prim array
  fiveDview grads;
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

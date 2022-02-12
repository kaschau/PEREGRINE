#ifndef __face__H__
#define __face__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct face_ {

  int _ng;
  // face number
  int _nface;
  // Boundary condition value arrays
  threeDview qBcVals, QBcVals;

  // MPI send and recv buffers
  threeDview sendBuffer3, recvBuffer3, tempRecvBuffer3;
  fourDview sendBuffer4, recvBuffer4, tempRecvBuffer4;


  // For cubic spline inlets
  fiveDviewHost cubicSplineAlphas;
  fourDview intervalAlphas;
  double intervalDt;
  int currentInterval = -1;

};

#endif

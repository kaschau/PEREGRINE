#ifndef __face__H__
#define __face__H__

#include "kokkosTypes.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data
// arrays for each block. Also converted into python class for modifying in the
// python wrapper
struct face_ {

  int _ng;
  // face number
  int _nface;
  // Boundary condition value arrays
  threeDview qBcVals, QBcVals;

  // MPI send and recv buffers
  // send
  threeDview sendBuffer_x, sendBuffer_y, sendBuffer_z;
  fourDview sendBuffer_q, sendBuffer_Q;
  fourDview sendBuffer_dqdx, sendBuffer_dqdy, sendBuffer_dqdz;
  fourDview sendBuffer_phi;
  // recv
  threeDview recvBuffer_x, recvBuffer_y, recvBuffer_z;
  fourDview recvBuffer_q, recvBuffer_Q;
  fourDview recvBuffer_dqdx, recvBuffer_dqdy, recvBuffer_dqdz;
  fourDview recvBuffer_phi;
  // temps
  threeDview tempRecvBuffer_x, tempRecvBuffer_y, tempRecvBuffer_z;
  fourDview tempRecvBuffer_q, tempRecvBuffer_Q;
  fourDview tempRecvBuffer_dqdx, tempRecvBuffer_dqdy, tempRecvBuffer_dqdz;
  fourDview tempRecvBuffer_phi;

  // For cubic spline inlets
  fiveDviewHost cubicSplineAlphas;
  fourDview intervalAlphas;
  double intervalDt;
  int currentInterval = -1;

  // Periodic rotation matricies
  twoDview periodicRotMatrixUp, periodicRotMatrixDown;
};

#endif

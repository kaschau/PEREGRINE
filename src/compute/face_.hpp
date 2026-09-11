#ifndef __face__H__
#define __face__H__

#include "kokkosTypes.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data
// arrays for each block. Also converted into python class for modifying in the
// python wrapper
struct face_ {

  // face number
  int nface;
  // Boundary condition value arrays
  threeDview qBcVals, QBcVals;

  // MPI send and recv buffers
  // send
  fourDview sendBuffer_nodes;
  fourDview sendBuffer_q, sendBuffer_Q;
  fiveDview sendBuffer_grads;
  fourDview sendBuffer_phi;
  // recv
  fourDview recvBuffer_nodes;
  fourDview recvBuffer_q, recvBuffer_Q;
  fiveDview recvBuffer_grads;
  fourDview recvBuffer_phi;
  // temps
  fourDview tempRecvBuffer_nodes;
  fourDview tempRecvBuffer_q, tempRecvBuffer_Q;
  fiveDview tempRecvBuffer_grads;
  fourDview tempRecvBuffer_phi;

  // How a halo arriving through this face is turned onto it
  twoDview periodicRotMatrix;
};

#endif

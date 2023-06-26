#ifndef __utils_H__
#define __utils_H__

#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "thtrdat_.hpp"

// ./utils
//    |------> applyFluxes
void applyFlux(block_ &b, thtrdat_ &th, double[]);
void applyHybridFlux(block_ &b, thtrdat_ &th, const double &primary);
void applyDissipationFlux(block_ &b, thtrdat_ &th, double[]);
//    |------> dQzero
void dQzero(block_ &b);
//    |------> dq2FD
void dq2FD(block_ &b);
void dq2FDoneSided(block_ &b, const int &nface);
//    |------> dq4FD
void dq4FD(block_ &b);
//    |------> axpby
void AEQConst(threeDview &A, const double &Const);
void AEQConst(fourDview &A, const double &Const);
void AEQB(fourDview &A, fourDview &B);
void ApEQxB(fourDview &A, const double &x, fourDview &B);
void AEQxB(fourDview &A, const double &x, fourDview &B);
void CEQxApyB(fourDview &C, const double &x, const fourDview &A,
              const double &y, const fourDview &B);
void CEQxApyB(threeDview &C, const double &x, const threeDview &A,
              const double &y, const threeDview &B);
std::array<double, 3> CFLmax(const std::vector<block_> &mb);
int checkNan(const std::vector<block_> &mb);

//    |------> sendRecvBuffer
void extractSendBuffer(threeDview &view, threeDview &buffer, face_ &face,
                       const std::vector<int> &slices);
void extractSendBuffer(fourDview &view, fourDview &buffer, face_ &face,
                       const std::vector<int> &slices);
void placeRecvBuffer(threeDview &view, threeDview &buffer, face_ &face,
                     const std::vector<int> &slices);
void placeRecvBuffer(fourDview &view, fourDview &buffer, face_ &face,
                     const std::vector<int> &slices);

//    |------> viscousSponge
void viscousSponge(block_ &b, const std::array<double, 3> &origin,
                   const std::array<double, 3> &ending, double mult);

double computeEntropy(const std::vector<block_> &mb, thtrdat_ &th);
double sumEntropy(const std::vector<block_> &mb);
void entropy(block_ &b, thtrdat_ &th);
#endif

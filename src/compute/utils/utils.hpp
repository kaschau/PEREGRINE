#ifndef __utils_H__
#define __utils_H__

#include "block_.hpp"
#include "compute.hpp"
#include <Kokkos_Core.hpp>

// ./utils
//    |------> applyFluxes
void applyFlux(block_ &b, double[]);
void applyHybridFlux(block_ &b, const double &primary);
void applyDissipationFlux(block_ &b, double[]);
//    |------> dQzero
void dQzero(block_ &b);
//    |------> dq2FD
void dq2FD(block_ &b);
void dq2FDoneSided(block_ &b, const int &nface);
//    |------> axpby
void AEQB(fourDview &A, fourDview &B);
// A = a*A + b*B [+ c*C], the linear combination every stage is built from
void axnpby(fourDview &A, const double &a, const double &b, const fourDview &B);
void axnpby(fourDview &A, const double &a, const double &b, const fourDview &B,
            const double &c, const fourDview &C);
std::array<double, 3> CFLmax(const std::vector<block_> &mb);
int checkNan(const std::vector<block_> &mb);

//    |------> sendRecvBuffer
void extractSendBuffer(fiveDview &view, fiveDview &buffer, face_ &face,
                       const std::vector<int> &slices);
void extractSendBuffer(fourDview &view, fourDview &buffer, face_ &face,
                       const std::vector<int> &slices);
void placeRecvBuffer(fiveDview &view, fiveDview &buffer, face_ &face,
                     const std::vector<int> &slices);
void placeRecvBuffer(fourDview &view, fourDview &buffer, face_ &face,
                     const std::vector<int> &slices);

//    |------> viscousSponge
void viscousSponge(block_ &b, const std::array<double, 3> &origin,
                   const std::array<double, 3> &ending, double mult);

#endif

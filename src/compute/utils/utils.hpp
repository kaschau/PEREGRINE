#ifndef __utils_H__
#define __utils_H__

#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"

// ./utils
//    |------> applyFluxes
void applyFlux(block_ b, double[]);
void applyHybridFlux(block_ b, const double primary);
void applyDissipationFlux(block_ b, double[]);
//    |------> dQzero
void dQzero(block_ b);
//    |------> dq2FD
void dq2FD(block_ b);
void dq2FDoneSided(block_ b, int nface);
//    |------> dq4FD
void dq4FD(block_ b);
//    |------> axpby
void AEQConst(fourDview A, const double Const);
void AEQB(fourDview A, fourDview B);
void ApEQxB(fourDview A, const double x, fourDview B);
void AEQxB(fourDview A, const double x, fourDview B);
void CEQxApyB(fourDview C, const double x, fourDview A, const double y,
              fourDview B);
std::array<double, 3> CFLmax(std::vector<block_> mb);
int checkNan(std::vector<block_> mb);

//    |------> sendRecvBuffer
void extract_sendBuffer3(threeDview view, face_ face, std::vector<int> slices);
void extract_sendBuffer4(fourDview view, face_ face, std::vector<int> slices);
void place_recvBuffer3(threeDview view, face_ face, std::vector<int> slices);
void place_recvBuffer4(fourDview view, face_ face, std::vector<int> slices);

#endif

#ifndef __compute_H__
#define __compute_H__

#include "Kokkos_Core.hpp"
#include "array"
#include "block_.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"
#include "vector"
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
// ./range_map.cpp
MDRange3 get_range3(block_ b, const int nface, const int indxI = 0,
                    const int indxJ = 0, const int indxK = 0);
threeDsubview getHaloSlice(fourDview view, const int nface, int slice);
twoDsubview getHaloSlice(threeDview view, const int nface, int slice);
void setHaloSlices(int &s0, int &s1, int &s2, int &plus, const int ni,
                   const int nj, const int nk, const int ng, const int nface);

// ./advFlux
//    |------> secondOrderKEEP
void secondOrderKEEP(block_ b, const thtrdat_ th);
//    |------> fourthOrderKEEP
void fourthOrderKEEP(block_ b, const thtrdat_ th);
//    |------> rusanov
void rusanov(block_ b, const thtrdat_ th);
//    |------> ausmPlusUp
void ausmPlusUp(block_ b, const thtrdat_ th);
//    |------> jamesonDissipation
void jamesonDissipation(block_ b, const thtrdat_ th);

// ./boundaryConditions
//    |------> inlets
void constantVelocitySubsonicInlet(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void supersonicInlet(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void constantMassFluxSubsonicInlet(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
//    |------> walls
void adiabaticNoSlipWall(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void adiabaticSlipWall(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void adiabaticMovingWall(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void isoTMovingWall(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
//    |------> exits
void constantPressureSubsonicExit(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);
void supersonicExit(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms);

// ./chemistry
//    |------> CH4_O2_Stanford_Skeletal
void chem_CH4_O2_Stanford_Skeletal(block_ b, const thtrdat_ th, const int face,
                                   const int indxI = 0, const int indxJ = 0,
                                   const int indxK = 0);
//    |------> GRI30
void chem_GRI30(block_ b, const thtrdat_ th, const int face,
                const int indxI = 0, const int indxJ = 0, const int indxK = 0);

// ./diffFlux
//    |------> diffusiveFlux
void diffusiveFlux(block_ b, const thtrdat_ th);

// ./switches
//    |------> jameson
void jamesonEntropy(block_ b);
void jamesonPressure(block_ b);
//    |------> vanAlbada
void vanAlbadaEntropy(block_ b);
void vanAlbadaPressure(block_ b);

// ./thermo
//    |------> cpg
void cpg(block_ b, const thtrdat_ th, const int face, const std::string given,
         const int indxI = 0, const int indxJ = 0, const int indxK = 0);
//    |------> tpg
void tpg(block_ b, const thtrdat_ th, const int face, const std::string given,
         const int indxI = 0, const int indxJ = 0, const int indxK = 0);
void cubic(block_ b, const thtrdat_ th, const int face, const std::string given,
           const int indxI = 0, const int indxJ = 0, const int indxK = 0);

// ./timeIntegration
//    |------> rk3Stages.cpp
void rk3s1(block_ b, const double dt);
void rk3s2(block_ b, const double dt);
void rk3s3(block_ b, const double dt);
//    |------> rk4Stages.cpp
void rk4s1(block_ b, const double dt);
void rk4s2(block_ b, const double dt);
void rk4s3(block_ b, const double dt);
void rk4s4(block_ b, const double dt);

// ./transport
//    |------> kineticThreory
void kineticTheory(block_ b, const thtrdat_ th, const int face,
                   const int indxI = 0, const int indxJ = 0,
                   const int indxK = 0);
//    |------> constantProps
void constantProps(block_ b, const thtrdat_ th, const int face,
                   const int indxI = 0, const int indxJ = 0,
                   const int indxK = 0);
//    |------> kineticThreoryUnityLewis
void kineticTheoryUnityLewis(block_ b, const thtrdat_ th, const int face,
                             const int indxI = 0, const int indxJ = 0,
                             const int indxK = 0);
//    |------> chungDenseGasUnityLewis
void chungDenseGasUnityLewis(block_ b, const thtrdat_ th, const int face,
                             const int indxI = 0, const int indxJ = 0,
                             const int indxK = 0);

// ./utils
//    |------> applyFluxes
void applyFlux(block_ b, double[]);
void applyHybridFlux(block_ b, const double primary);
void applyDissipationFlux(block_ b, double[]);
//    |------> dQzero
void dQzero(block_ b);
//    |------> dq2FD
void dq2FD(block_ b);
//    |------> dq4FD
void dq4FD(block_ b);
//    |------> axpby
void AEQB(fourDview A, fourDview B);
void ApEQxB(fourDview A, const double x, fourDview B);
void AEQxB(fourDview A, const double x, fourDview B);
void CEQxApyB(fourDview C, const double x, fourDview A, const double y,
              fourDview B);
std::array<double, 2> CFLmax(std::vector<block_> mb);
int checkNan(std::vector<block_> mb);

#endif

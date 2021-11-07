#ifndef __compute_H__
#define __compute_H__

#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "face_.hpp"
#include "thtrdat_.hpp"
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
// ./range_map.cpp
MDRange3 get_range3(block_ b, const int nface, const int i=0, const int j=0, const int k=0);
threeDsubview getHaloSlice(fourDview view, const int nface, int slice);
void setHaloSlices(int& s0, int& s1, int& s2, int& plus, const int ni, const int nj, const int nk, const int ng, const int nface);

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
void constantVelocitySubsonicInlet(block_ b, const face_ th);


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
//    |------> negateFluxes
void noIFlux(block_ b);
void noJFlux(block_ b);
void noKFlux(block_ b);
void noInoJFlux(block_ b);
void noInoKFlux(block_ b);
void noJnoKFlux(block_ b);


// ./thermo
//    |------> cpg
void cpg(block_ b,
         const thtrdat_ th,
         const int face,
         const std::string given,
         const int i=0,
         const int j=0,
         const int k=0);
//    |------> tpg
void tpg(block_ b,
         const thtrdat_ th,
         const int face,
         const std::string given,
         const int i=0,
         const int j=0,
         const int k=0);


// ./transport
//    |------> kineticThreory
void kineticTheory(block_ b,
                   const thtrdat_ th,
                   const int face,
                   const int i=0,
                   const int j=0,
                   const int k=0);
//    |------> constantProps
void constantProps(block_ b,
                   const thtrdat_ th,
                   const int face,
                   const int i=0,
                   const int j=0,
                   const int k=0);


// ./chemistry
//    |------> CH4_O2_Stanford_Skeletal
void chem_CH4_O2_Stanford_Skeletal(block_ b,
                             const thtrdat_ th,
                             const int face,
                             const int i=0,
                             const int j=0,
                             const int k=0);
//    |------> GRI30
void chem_GRI30(block_ b,
          const thtrdat_ th,
          const int face,
          const int i=0,
          const int j=0,
          const int k=0);


// ./utils
//    |------> applyFluxes
void applyFlux(block_ b, const double primary);
void applyHybridFlux(block_ b, const double primary);
void applyDissipationFlux(block_ b, const double primary);
//    |------> dQzero
void dQzero(block_ b);
//    |------> dq2FD
void dq2FD(block_ b);
//    |------> dq4FD
void dq4FD(block_ b);

#endif

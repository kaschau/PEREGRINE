#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
MDRange3 get_range3(block_ b, int face, int i=0, int j=0, int k=0);

// ./advFlux
//    |------> centralEuler
void centralEuler(block_ b, const thtrdat_ th, const double primary);
//    |------> rusanov
void rusanov(block_ b, const thtrdat_ th, const double primary);
//    |------> ausmPlusUp
void ausmPlusUp(block_ b, const thtrdat_ th, const double primary);


// ./diffFlux
//    |------> centralVisc
void centralVisc(block_ b, const thtrdat_ th);


// ./switches
//    |------> pressure
void pressure(block_ b);


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
//    |------> transport
void kineticTheory(block_ b,
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
//    |------> dQzero
void dQzero(block_ b);
//    |------> dq2FD
void dq2FD(block_ b);

#endif

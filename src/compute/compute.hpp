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
void centralEuler(block_ b, thtrdat_ th, double primary);


// ./diffFlux
//    |------> centralVisc
void centralVisc(block_ b, thtrdat_ th);


// ./thermo
//    |------> cpg
void cpg(block_ b, thtrdat_ th, int face, std::string given, int i=0, int j=0, int k=0);
//    |------> tpg
void tpg(block_ b, thtrdat_ th, int face, std::string given, int i=0, int j=0, int k=0);


// ./transport
//    |------> transport
void kineticTheory(block_ b, thtrdat_ th, int face, int i=0, int j=0, int k=0);


// ./chemistry
//    |------> CH4_O2_Stanford_Skeletal
void chem_CH4_O2_Stanford_Skeletal(block_ b, thtrdat_ th, int face=0, int i=0, int j=0, int k=0);
//    |------> GRI30
void chem_GRI30(block_ b, thtrdat_ th, int face=0, int i=0, int j=0, int k=0);


// ./utils
//    |------> dQzero
void dQzero(block_ b);
//    |------> dq2FD
void dq2FD(block_ b);

#endif

#ifndef __compute_H__
#define __compute_H__

#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"
#include <vector>
#include <string>

///////////////////////////////////////////////////////////
////////////////// Compute Functions //////////////////////
///////////////////////////////////////////////////////////
MDRange3 get_range3(block_ b, int face);

// ./flux
//    |------> dQzero
void dQzero(std::vector<block_> mb);
//    |------> dqdxyz
void dqdxyz(std::vector<block_> mb);
//    |------> Advective
void advective(std::vector<block_> mb, thtrdat_ th);
//    |------> Diffusive
void diffusive(std::vector<block_> mb, thtrdat_ th);

// ./thermo
//    |------> cpg
void cpg(block_ b, thtrdat_ th, int face, std::string given);
//    |------> tpg
void tpg(block_ b, thtrdat_ th, int face, std::string given);

#endif

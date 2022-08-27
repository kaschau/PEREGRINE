#ifndef __thermo_H__
#define __thermo_H__

#include "block_.hpp"
#include "thtrdat_.hpp"

// ./thermo
//    |------> cpg
void cpg(block_ &b, const thtrdat_ &th, const int &face,
         const std::string &given, const int &indxI /*=0*/,
         const int &indxJ /*=0*/, const int &indxK /*=0*/);
//    |------> tpg
void tpg(block_ &b, const thtrdat_ &th, const int &face,
         const std::string &given, const int &indxI /*=0*/,
         const int &indxJ /*=0*/, const int &indxK /*=0*/);
//    |------> cubic
void cubic(block_ &b, const thtrdat_ &th, const int &face,
           const std::string &given, const int &indxI /*=0*/,
           const int &indxJ /*=0*/, const int &indxK /*=0*/);

#endif

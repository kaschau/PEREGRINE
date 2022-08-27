#ifndef __transport_H__
#define __transport_H__

#include "block_.hpp"
#include "thtrdat_.hpp"

// ./transport
//    |------> kineticThreory
void kineticTheory(block_ &b, const thtrdat_ &th, const int &face,
                   const int &indxI /*=0*/, const int &indxJ /*=0*/,
                   const int &indxK /*=0*/);
//    |------> constantProps
void constantProps(block_ &b, const thtrdat_ &th, const int &face,
                   const int &indxI /*=0*/, const int &indxJ /*=0*/,
                   const int &indxK /*=0*/);
//    |------> kineticThreoryUnityLewis
void kineticTheoryUnityLewis(block_ &b, const thtrdat_ &th, const int &face,
                             const int &indxI /*=0*/, const int &indxJ /*=0*/,
                             const int &indxK /*=0*/);
//    |------> chungDenseGasUnityLewis
void chungDenseGasUnityLewis(block_ &b, const thtrdat_ &th, const int &face,
                             const int &indxI /*=0*/, const int &indxJ /*=0*/,
                             const int &indxK /*=0*/);

#endif

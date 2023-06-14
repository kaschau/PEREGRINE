#ifndef __advFlux_H__
#define __advFlux_H__

#include "block_.hpp"
#include "thtrdat_.hpp"

// ./advFlux
//    |------> secondOrderKEEP
void secondOrderKEEP(block_ &b, const thtrdat_ &th);
//    |------> myKEEP
void myKEEP(block_ &b, const thtrdat_ &th);
//    |------> roeEC
void roeEC(block_ &b, const thtrdat_ &th);
//    |------> centralDifference
void centralDifference(block_ &b, const thtrdat_ &th);
//    |------> fourthOrderKEEP
void fourthOrderKEEP(block_ &b);
//    |------> scalarDissipation
void scalarDissipation(block_ &b);
//    |------> rusanov
void rusanov(block_ &b);
//    |------> ausmPlusUp
void ausmPlusUp(block_ &b);
//    |------> hllc
void hllc(block_ &b);
//    |------> muscl2hllc
void muscl2hllc(block_ &b);
//    |------> muscl2rusanov
void muscl2rusanov(block_ &b);
//    |------> KEEPdissipation
void KEEPdissipation(block_ &b);

#endif

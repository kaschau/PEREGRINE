#ifndef __advFlux_H__
#define __advFlux_H__

#include "block_.hpp"

// ./advFlux
//    |------> secondOrderKEEP
void secondOrderKEEP(block_ &b);
//    |------> myKEEP
void myKEEP(block_ &b);
//    |------> centralDifference
void centralDifference(block_ &b);
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

#ifndef __advFlux_H__
#define __advFlux_H__

#include "block_.hpp"

// ./advFlux
//    |------> KEEP
void KEEP(block_ &b);
//    |------> KEEPpe
void KEEPpe(block_ &b);
//    |------> KEPaEC
void KEPaEC(block_ &b);
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

#endif

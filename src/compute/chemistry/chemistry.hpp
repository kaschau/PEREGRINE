#ifndef __chemistry_H__
#define __chemistry_H__

#include "block_.hpp"
#include "face_.hpp"
#include "thtrdat_.hpp"

// ./chemistry
//    |------> C2H4_Air_Red22
void chem_C2H4_Air_Red22(block_ &b, const thtrdat_ &th, const int &rface,
                         const int &indxI = 0, const int &indxJ = 0,
                         const int &indxK = 0, const int &nChemSubSteps = 1,
                         const double &dt = 1.0);
//    |------> C2H4_Air_Skeletal
void chem_C2H4_Air_Skeletal(block_ &b, const thtrdat_ &th, const int &rface,
                            const int &indxI = 0, const int &indxJ = 0,
                            const int &indxK = 0, const int &nChemSubSteps = 1,
                            const double &dt = 1.0);
//    |------> CH4_O2_FFCMY
void chem_CH4_O2_FFCMY(block_ &b, const thtrdat_ &th, const int &rface,
                       const int &indxI = 0, const int &indxJ = 0,
                       const int &indxK = 0, const int &nChemSubSteps = 1,
                       const double &dt = 1.0);
//    |------> GRI30
void chem_GRI30(block_ &b, const thtrdat_ &th, const int &rface,
                const int &indxI = 0, const int &indxJ = 0,
                const int &indxK = 0, const int &nChemSubSteps = 1,
                const double &dt = 1.0);

#endif

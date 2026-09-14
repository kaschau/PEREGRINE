#include "boundaryConditions/faceRecords.hpp"
#include "kernelUtils.hpp"

// One condition at one hook, over every face that has it. The jit compiles
// this once per condition and hook: the condition's header is the forced
// include, and PG_CONDITION and PG_HOOK name the function in it.
#ifndef PG_CONDITION
#error                                                                         \
    "a hook is compiled for one condition: -DPG_CONDITION and -DPG_HOOK from the jit"
#endif
#define PG_PASTE_(a, b) a##_##b
#define PG_PASTE(a, b) PG_PASTE_(a, b)

PG_ABI void pgHook(int count, pgOut *q_, pgOut *Q_, pgIn *qh_, pgOut *grads_,
                   pgIn *S_, pgIn *qBcVals_, pgIn *QBcVals_, pgIn *rot_,
                   const int *nface, double tme) {
  for (int e = 0; e < count; e++) {
    const faceRecords face{as4(q_[e]),       as4(Q_[e]),     as4(qh_[e]),
                           as4(S_[e]),       as5(grads_[e]), as3(qBcVals_[e]),
                           as3(QBcVals_[e]), as2(rot_[e]),   nface[e]};
    PG_PASTE(PG_CONDITION, PG_HOOK)(face, tme);
  }
}

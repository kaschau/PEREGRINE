#include "kernel.hpp"

// One condition at one hook, over every face that has it, in one launch.
// The jit compiles this once per condition and hook: the condition's header
// is the forced include, and PG_CONDITION and PG_HOOK name the struct in it.
#ifndef PG_CONDITION
#error                                                                         \
    "a hook is compiled for one condition: -DPG_CONDITION and -DPG_HOOK from the jit"
#endif
#define PG_STRING_(a) #a
#define PG_STRING(a) PG_STRING_(a)

using condition = PG_CONDITION::PG_HOOK;

PG_RANGE(facePlanes, ng)
PG_ABI void pgHook(const condition &c, const pgTiling &t, const int *nface) {
  forFacePlanes(PG_STRING(PG_CONDITION) " " PG_STRING(PG_HOOK), t, c, nface);
}

#if !defined(DEXRT_INITIAL_POPS_HPP)
#define DEXRT_INITIAL_POPS_HPP

/**
 * Sets the initial populations if something _other_ than LTE is requested, e.g. ZeroRadiation
 */
template <typename State>
void set_initial_pops_special(State* state);

#else
#endif

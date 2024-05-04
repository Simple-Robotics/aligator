#pragma once

#if __cplusplus >= 201703L
#define ALIGATOR_WITH_CPP_17
#endif

#if __cplusplus >= 201402L
#define ALIGATOR_WITH_CPP_14
#endif

#define ALIGATOR_INLINE inline __attribute__((always_inline))

#ifdef ALIGATOR_WITH_CPP_17
#define ALIGATOR_MAYBE_UNUSED [[maybe_unused]]
#elif defined(_MSC_VER) && !defined(__clang__)
#define ALIGATOR_MAYBE_UNUSED
#else
#define ALIGATOR_MAYBE_UNUSED __attribute__((__unused__))
#endif

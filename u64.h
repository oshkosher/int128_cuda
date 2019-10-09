#ifndef __U64_H__
#define __U64_H__

#include <string>
#include <limits.h>

#if ULLONG_MAX != 18446744073709551615ULL
#error Code assumes sizeof(long long unsigned)==8
#endif


typedef long long unsigned u64;

#ifndef PRIu64
#define PRIu64 "llu"
#define PRIx64 "llx"
#endif

// returns false on failure
bool stringToU64(u64 &result, const char *str);
std::string u64ToString(u64 x);
std::string commafy(u64 x);

#endif // __U64_H__

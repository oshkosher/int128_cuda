#include <string>
#include <stdlib.h>
#include <stdio.h>
#include "int128.h"
#include "fixed_int.h"
// #include "myassert.cc"

using namespace std;

/*
see
http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda
*/

// define reverse versions of the overloaded operators
/*
HD Int128 operator * (u64 x, Int128 o) {return o * x;}
HD Int128 operator + (u64 x, Int128 o) {return o + x;}
HD Int128 operator - (u64 x, Int128 o) {return o - x;}
HD bool operator == (u64 x, Int128 o) {return o == x;}
HD bool operator != (u64 x, Int128 o) {return o != x;}
*/

// enable use of Int128 objects in ostreams
std::ostream& operator << (std::ostream &o, Int128 x) {
  /*
  string base10;
  if (x == 0) {
    base10 = "0";
  } else {
    Int128 tmp_x = x;
    while (tmp_x != 0) {
      unsigned m;
      tmp_x.divMod(10, tmp_x, m);
      base10 += (char)('0' + m);
    }
    base10 = string(base10.rbegin(), base10.rend());
  }
  */
  return o << "0x" << std::hex << std::setfill('0')
	   << std::setw(8) << x.word3() << ' '
	   << std::setw(8) << x.word2() << ' '
	   << std::setw(8) << x.word1() << ' '
	   << std::setw(8) << x.word0()
    // << "(" << base10 << ")"
	   << std::dec << std::setfill(' ');

}

#ifndef __INT_128__
#define __INT_128__

/*
  128-bit extra long integer class
  Using four 32-bit integers internally.

  Ed Karrels, May 2012
*/

// http://tauday.com/
// http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda
// http://stackoverflow.com/questions/6659414/efficient-128-bit-addition-using-carry-flag?rq=1
// http://software.intel.com/en-us/forums/topic/278561

#include <iostream>
#include <iomanip>
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include "cudafree.h"
#include "xmp.h"

using namespace std;

#if UINT_MAX != 4294967295
#error Assume 32 bit unsigned integer
#endif

#define LO_MASK 0x00000000FFFFFFFFLL
#define HI_MASK 0xFFFFFFFF00000000LL

typedef long long unsigned u64;

class Int128; // forward declaration

std::ostream& operator << (std::ostream &o, Int128 x);


class Int128 {

 public: 

#if NVCC_VERSION == 50
  // for some reason, this slows down a bit with the 5.0 compiler
  // if I remove the union
  union {
    struct {
      u64 unused0, unused1;
    };
    struct {
      unsigned dataword0, dataword1, dataword2, dataword3;
    };
  };
#else
  unsigned dataword0, dataword1, dataword2, dataword3;
#endif
  
  HD Int128() {}

  HD Int128(u64 init) {
    setPart0(init);
    setPart1(0);
  }

  HD Int128(u64 hi, u64 lo) {
    setPart0(lo);
    setPart1(hi);
  }

  HD Int128(unsigned w3, unsigned w2, unsigned w1, unsigned w0)
    : dataword0(w0), dataword1(w1), dataword2(w2), dataword3(w3) {}

  HD Int128(const Int128 &other)
    : dataword0(other.dataword0), dataword1(other.dataword1),
      dataword2(other.dataword2), dataword3(other.dataword3) {}

  HD u64 getPart1() const {return XMP::join64(dataword3, dataword2);}
  HD u64 getPart0() const {return XMP::join64(dataword1, dataword0);}

  HD void setPart1(u64 x) {
    unsigned hi, lo;
    XMP::split64(x, hi, lo);
    word2() = lo;
    word3() = hi;
  }
  HD void setPart0(u64 x) {
    unsigned hi, lo;
    XMP::split64(x, hi, lo);
    word0() = lo;
    word1() = hi;
  }

  HD u64 to_u64() const {return getPart0();}

  HD unsigned& word0() {return dataword0;}
  HD void setWord0(unsigned x) {dataword0 = x;}

  HD unsigned& word1() {return dataword1;}
  HD void setWord1(unsigned x) {dataword1 = x;}

  HD unsigned& word2() {return dataword2;}
  HD void setWord2(unsigned x) {dataword2 = x;}

  HD unsigned& word3() {return dataword3;}
  HD void setWord3(unsigned x) {dataword3 = x;}


  bool from_hex(const char *str) {
    // skip leading whitespace
    while (isspace(*str)) str++;

    // skip optional "0x" prefix
    if (str[0] == '0' && tolower(str[1]) == 'x') str += 2;

    if (!isxdigit(*str)) return false;
    
    Int128 result = 0;
    
    for (; isxdigit(*str); str++) {
      int digit;
      if (isdigit(*str)) {
	digit = *str - '0';
      } else {
	digit = tolower(*str) - 'a' + 10;
      }
      result = result * 16 + digit;
    }

    *this = result;

    return true;
  }

  HD const char *toString(char buf[33]) {
    XMP::toHex(buf, word3());
    XMP::toHex(buf+8, word2());
    XMP::toHex(buf+16, word1());
    XMP::toHex(buf+24, word0());
    buf[32] = '\0';
    return buf;
  }
	

  // overload equality
  HD bool operator == (Int128 x) {
    return word0() == x.word0() &&
           word1() == x.word1() &&
           word2() == x.word2() &&
           word3() == x.word3();
  }

  HD bool operator == (u64 x) {
    return getPart0() == x && word2() == 0 && word3() == 0;
  }

  HD bool operator != (Int128 x) {
    return word0() != x.word0() ||
           word1() != x.word1() ||
           word2() != x.word2() ||
           word3() != x.word3();
  }

  HD bool operator != (u64 x) {
    return getPart0() != x || word2() != 0 || word3() != 0;
  }

  HD bool operator < (Int128 x) {
    if (word3() == x.word3()) {
      if (word2() == x.word2()) {
	if (word1() == x.word1()) {
	  return word0() < x.word0();
	} else {
	  return word1() < x.word1();
	}
      } else {
	return word2() < x.word2();
      }
    } else {
      return word3() < x.word3();
    }
  }

  HD bool operator > (Int128 that) {
    return that < *this;
  }

  HD bool operator <= (Int128 that) {
    return !(that < *this);
  }

  HD bool operator >= (Int128 that) {
    return !(*this < that);
  }


  // cast to double - more trouble than it's worth
  HD double toDouble() {
    double result = (double)getPart0();
    if (getPart1()) {
      // 1<<64
      // result += 18446744073709551616.0 * getPart1();
      result += (double)ULLONG_MAX * getPart1();
    }
    return result;
  }

  HD double toFraction() {
    return toDouble() / 3.4028236692093846e+38;
  }


  // overload shift operators

  HD Int128 operator << (unsigned s) {
    Int128 result;
    s &= 127;  // modulo 128
    if (s == 0) return *this;
    if (s >= 64) {
      result.setPart0(0);
      result.setPart1(getPart0() << (s-64));
    } else {
      result.setPart0(getPart0() << s);
      result.setPart1((getPart1() << s) | (getPart0() >> (64-s)));
    }
    return result;
  }

  HD Int128 operator >> (unsigned s) {
    Int128 result;
    s &= 127;  // modulo 128
    if (s == 0) return *this;
    if (s >= 64) {
      result.setPart1(0);
      result.setPart0(getPart1() >> (s-64));
    } else {
      result.setPart1(getPart1() >> s);
      result.setPart0((getPart0() >> s) | (getPart1() << (64-s)));
    }
    return result;
  }

  HD Int128 operator >>= (unsigned s) {
    return *this = *this >> s;
  }

  HD Int128 operator <<= (unsigned s) {
    return *this = *this << s;
  }

  // overload bitwise operators

  HD Int128 operator | (Int128 x) {
    return Int128(word3()|x.word3(), word2()|x.word2(), word1()|x.word1(), 
		  word0()|x.word0());
  }

  HD Int128 operator & (Int128 x) {
    return Int128(word3()&x.word3(), word2()&x.word2(), word1()&x.word1(), 
		  word0()&x.word0());
  }

  HD Int128 operator &= (Int128 x) {
    return *this = *this & x;
  }

  HD Int128 operator ~ () {
    return Int128(~word3(), ~word2(), ~word1(), ~word0());
  }

  HD Int128 operator - () {
    return ~(*this) + 1;
  }

  HD int countLeadingZeros() {
    if (word3()) {
      return XMP::countLeadingZeros(word3());
    } else if (word2()) {
      return 32+XMP::countLeadingZeros(word2());
    } else if (word1()) {
      return 64+XMP::countLeadingZeros(word1());
    } else {
      return 96+XMP::countLeadingZeros(word0());
    }
  }

  HD float toFloat() {
    return word3() * 79228162514264337593543950336.0f + // 2^96
      word2() * 18446744073709551616.0f + // 2^64
      word1() * 4294967296.0f + // 2^32
      word0();
  }

#ifdef __CUDACC__
  __device__ static Int128 add_128_128_asm(Int128 a, Int128 b) {
    Int128 result;
    asm("{\n\t"
	"add.cc.u32    %0, %4, %8;\n\t"
	"addc.cc.u32   %1, %5, %9;\n\t"
	"addc.cc.u32   %2, %6, %10;\n\t"
	"addc.u32      %3, %7, %11;\n\t"
	"}\n"
        : "=r"(result.word0()), "=r"(result.word1()),
	  "=r"(result.word2()), "=r"(result.word3())
	: "r"(a.word0()), "r"(a.word1()),
	  "r"(a.word2()), "r"(a.word3()),
	  "r"(b.word0()), "r"(b.word1()),
	  "r"(b.word2()), "r"(b.word3())
	);
    return result;
  }

  __device__ static Int128 add_128_64_asm(Int128 a, u64 b) {
    Int128 result;
    asm("{\n\t"
	".reg .u32 sb<2>;\n\t"
	"mov.b64   {sb0,sb1}, %8;\n\t"
	"add.cc.u32    %0, %4, sb0;\n\t"
	"addc.cc.u32   %1, %5, sb1;\n\t"
	"addc.cc.u32   %2, %6, 0;\n\t"
	"addc.u32      %3, %7, 0;\n\t"
	"}\n"
        : "=r"(result.word0()), "=r"(result.word1()),
	  "=r"(result.word2()), "=r"(result.word3())
	: "r"(a.word0()), "r"(a.word1()),
	  "r"(a.word2()), "r"(a.word3()),
	  "l"(b)
	);
    return result;
  }

  __device__ static Int128 sub_128_128_asm(Int128 a, Int128 b) {
    Int128 result;
    asm("{\n\t"
	"sub.cc.u32    %0, %4, %8;\n\t"
	"subc.cc.u32   %1, %5, %9;\n\t"
	"subc.cc.u32   %2, %6, %10;\n\t"
	"subc.u32      %3, %7, %11;\n\t"
	"}\n"
        : "=r"(result.word0()), "=r"(result.word1()),
	  "=r"(result.word2()), "=r"(result.word3())
	: "r"(a.word0()), "r"(a.word1()),
	  "r"(a.word2()), "r"(a.word3()),
	  "r"(b.word0()), "r"(b.word1()),
	  "r"(b.word2()), "r"(b.word3())
	);
    return result;
  }


  __device__ static Int128 sub_128_64_asm(Int128 a, u64 b) {
    Int128 result;
    asm("{\n\t"
	".reg .u32 sb<2>;\n\t"
	"mov.b64   {sb0,sb1}, %8;\n\t"
	"sub.cc.u32    %0, %4, sb0;\n\t"
	"subc.cc.u32   %1, %5, sb1;\n\t"
	"subc.cc.u32   %2, %6, 0;\n\t"
	"subc.u32      %3, %7, 0;\n\t"
	"}\n"
        : "=r"(result.word0()), "=r"(result.word1()),
	  "=r"(result.word2()), "=r"(result.word3())
	: "r"(a.word0()), "r"(a.word1()),
	  "r"(a.word2()), "r"(a.word3()),
	  "l"(b)
	);
    return result;
  }

  __device__ static void mult128_asm(Int128 a, Int128 b,
				     Int128 &result_hi, Int128 &result_lo) {
    // write_mult_asm.py 2 4 full 'a.word%d()' 'b.word%d()' 'result_lo.word%d(),result_hi.word%d()'
    asm( "{\n\t"
	 ".reg .u32 tmp;\n\t"
	 "mul.lo.u32       %0, %8, %12;\n\t"
	 "mul.lo.u32       %1, %8, %13;\n\t"
	 "mul.lo.u32       %2, %8, %14;\n\t"
	 "mul.lo.u32       %3, %8, %15;\n\t"

	 "mul.hi.u32       tmp, %8, %12;\n\t"
	 "add.cc.u32       %1, %1, tmp;\n\t"
	 "mul.hi.u32       tmp, %8, %13;\n\t"
	 "addc.cc.u32      %2, %2, tmp;\n\t"
	 "mul.hi.u32       tmp, %8, %14;\n\t"
	 "addc.cc.u32      %3, %3, tmp;\n\t"
	 "mul.hi.u32       tmp, %8, %15;\n\t"
	 "addc.u32         %4, 0, tmp;\n\t"

	 "mul.lo.u32       tmp, %9, %12;\n\t"
	 "add.cc.u32       %1, %1, tmp;\n\t"
	 "mul.lo.u32       tmp, %9, %13;\n\t"
	 "addc.cc.u32      %2, %2, tmp;\n\t"
	 "mul.lo.u32       tmp, %9, %14;\n\t"
	 "addc.cc.u32      %3, %3, tmp;\n\t"
	 "mul.lo.u32       tmp, %9, %15;\n\t"
	 "addc.u32         %4, %4, tmp;\n\t"

	 "mul.hi.u32       tmp, %9, %12;\n\t"
	 "add.cc.u32       %2, %2, tmp;\n\t"
	 "mul.hi.u32       tmp, %9, %13;\n\t"
	 "addc.cc.u32      %3, %3, tmp;\n\t"
	 "mul.hi.u32       tmp, %9, %14;\n\t"
	 "addc.cc.u32      %4, %4, tmp;\n\t"
	 "mul.hi.u32       tmp, %9, %15;\n\t"
	 "addc.u32         %5, 0, tmp;\n\t"

	 "mul.lo.u32       tmp, %10, %12;\n\t"
	 "add.cc.u32       %2, %2, tmp;\n\t"
	 "mul.lo.u32       tmp, %10, %13;\n\t"
	 "addc.cc.u32      %3, %3, tmp;\n\t"
	 "mul.lo.u32       tmp, %10, %14;\n\t"
	 "addc.cc.u32      %4, %4, tmp;\n\t"
	 "mul.lo.u32       tmp, %10, %15;\n\t"
	 "addc.u32         %5, %5, tmp;\n\t"

	 "mul.hi.u32       tmp, %10, %12;\n\t"
	 "add.cc.u32       %3, %3, tmp;\n\t"
	 "mul.hi.u32       tmp, %10, %13;\n\t"
	 "addc.cc.u32      %4, %4, tmp;\n\t"
	 "mul.hi.u32       tmp, %10, %14;\n\t"
	 "addc.cc.u32      %5, %5, tmp;\n\t"
	 "mul.hi.u32       tmp, %10, %15;\n\t"
	 "addc.u32         %6, 0, tmp;\n\t"

	 "mul.lo.u32       tmp, %11, %12;\n\t"
	 "add.cc.u32       %3, %3, tmp;\n\t"
	 "mul.lo.u32       tmp, %11, %13;\n\t"
	 "addc.cc.u32      %4, %4, tmp;\n\t"
	 "mul.lo.u32       tmp, %11, %14;\n\t"
	 "addc.cc.u32      %5, %5, tmp;\n\t"
	 "mul.lo.u32       tmp, %11, %15;\n\t"
	 "addc.u32         %6, %6, tmp;\n\t"

	 "mul.hi.u32       tmp, %11, %12;\n\t"
	 "add.cc.u32       %4, %4, tmp;\n\t"
	 "mul.hi.u32       tmp, %11, %13;\n\t"
	 "addc.cc.u32      %5, %5, tmp;\n\t"
	 "mul.hi.u32       tmp, %11, %14;\n\t"
	 "addc.cc.u32      %6, %6, tmp;\n\t"
	 "mul.hi.u32       tmp, %11, %15;\n\t"
	 "addc.u32         %7, 0, tmp;\n\t"

	 "}\n\t"
	 : "=r"(result_lo.word0()), "=r"(result_lo.word1()), "=r"(result_lo.word2()), "=r"(result_lo.word3()), "=r"(result_hi.word0()), "=r"(result_hi.word1()), "=r"(result_hi.word2()), "=r"(result_hi.word3())
	 : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
	   "r"(b.word0()), "r"(b.word1()), "r"(b.word2()), "r"(b.word3())
	 );

  }

  __device__ static int atomicAddWithCarry(unsigned *result, unsigned value,
				    int carryIn) {
    unsigned old = ::atomicAdd(result, value + carryIn);
    unsigned sum = old + value + carryIn;
    if (carryIn)
      return sum <= value;
    else
      return sum < value;
  }

  __device__ void atomicAdd(Int128 value) {
    int carry;
    carry = atomicAddWithCarry(&dataword0, value.word0(), 0);
    carry = atomicAddWithCarry(&dataword1, value.word1(), carry);
    carry = atomicAddWithCarry(&dataword2, value.word2(), carry);
    atomicAddWithCarry(&dataword3, value.word3(), carry);
  }
  
#endif


  HD Int128 operator + (Int128 x) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    Int128 result;
    int carry;
    result.word0() = XMP::addCarryOut  (word0(), x.word0(), carry);
    result.word1() = XMP::addCarryInOut(word1(), x.word1(), carry);
    result.word2() = XMP::addCarryInOut(word2(), x.word2(), carry);
    result.word3() = XMP::addCarryIn   (word3(), x.word3(), carry);
    return result;
#else
    return add_128_128_asm(*this, x);
#endif
  }

  HD Int128 operator += (u64 x) {
    return *this = *this + x;
  }

  HD Int128 operator += (const Int128 x) {
    return *this = *this + x;
  }

  // overload subtraction operator
  HD Int128 operator - (Int128 x) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    Int128 result;
    int borrow;
    result.word0() = XMP::subBorrowOut  (word0(), x.word0(), borrow);
    result.word1() = XMP::subBorrowInOut(word1(), x.word1(), borrow);
    result.word2() = XMP::subBorrowInOut(word2(), x.word2(), borrow);
    result.word3() = XMP::subBorrowIn   (word3(), x.word3(), borrow);
    return result;
#else
    return sub_128_128_asm(*this, x);
#endif
  }

  HD Int128 operator -= (const Int128 x) {
    return *this = *this - x;
  }



  // multiplication
  // variations included:
  // 64*64 -> 128, hi(64*64)  (lo not necessary; built-in)
  // 128*64 -> 128
  // 128*128 -> 256, hi(128*128), lo(128*128)


  HD static Int128 mult64(u64 a, u64 b) {
    u64 tmpHi, tmpLo;
    XMP::multFull64(a, b, tmpHi, tmpLo);
    return Int128(tmpHi, tmpLo);
  }

  /*
    Full 128-bit multiply -> 256 bit result
           A B
           C D
        ------
          --BD
        --AD
        --BC
      --AC
  */

  HD static void mult128(Int128 a, Int128 b,
			 Int128 &result_hi, Int128 &result_lo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    int carry;
    // --BD
    result_lo = mult64(a.getPart0(), b.getPart0());

    // --AC
    result_hi = mult64(a.getPart1(), b.getPart1());

    // --AD
    Int128 tmp = mult64(a.getPart1(), b.getPart0());
    result_lo.setPart1(XMP::addCarryOut64
		       (result_lo.getPart1(), tmp.getPart0(), carry));
    result_hi.setPart0(XMP::addCarryInOut64
		       (result_hi.getPart0(), tmp.getPart1(), carry));
    result_hi.setPart1(XMP::addCarryIn64
		       (result_hi.getPart1(), 0, carry));

    // --BC
    tmp = mult64(a.getPart0(), b.getPart1());
    result_lo.setPart1(XMP::addCarryOut64
		       (result_lo.getPart1(), tmp.getPart0(), carry));
    result_hi.setPart0(XMP::addCarryInOut64
		       (result_hi.getPart0(), tmp.getPart1(), carry));
    result_hi.setPart1(XMP::addCarryIn64
		       (result_hi.getPart1(), 0, carry));

#else

    mult128_asm(a, b, result_hi, result_lo);

#endif
  }

  HD static Int128 mult128hi(Int128 a, Int128 b) {
    Int128 result_lo, result_hi;
    mult128(a, b, result_hi, result_lo);
    return result_hi;
  }

  HD static Int128 mult128lo(Int128 a, Int128 b) {
    Int128 result;

#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    result.setPart0(a.getPart0() * b.getPart0());
    result.setPart1(XMP::multHi64(a.getPart0(), b.getPart0()) +
		    a.getPart0() * b.getPart1() +
		    a.getPart1() * b.getPart0());
#else
    result = mult128lo_asm(a, b);
#endif

    return result;
  }

#ifdef __CUDACC__
  __device__ static Int128 mult128lo_asm(Int128 a, Int128 b) {
    /*
    // write_mult_asm.py 2 4 lo 'a.word%d()' 'b.word%d()' 'result.word%d()'
    Int128 result;
    asm( "{\n\t"
	 ".reg .u32 tmp;\n\t"
	 "mul.lo.u32       %0, %4, %8;\n\t"
	 "mul.lo.u32       %1, %4, %9;\n\t"
	 "mul.lo.u32       %2, %4, %10;\n\t"
	 "mul.lo.u32       %3, %4, %11;\n\t"

	 "mul.hi.u32       tmp, %4, %8;\n\t"
	 "add.cc.u32       %1, %1, tmp;\n\t"
	 "mul.hi.u32       tmp, %4, %9;\n\t"
	 "addc.cc.u32      %2, %2, tmp;\n\t"
	 "mul.hi.u32       tmp, %4, %10;\n\t"
	 "addc.u32         %3, %3, tmp;\n\t"

	 "mul.lo.u32       tmp, %5, %8;\n\t"
	 "add.cc.u32       %1, %1, tmp;\n\t"
	 "mul.lo.u32       tmp, %5, %9;\n\t"
	 "addc.cc.u32      %2, %2, tmp;\n\t"
	 "mul.lo.u32       tmp, %5, %10;\n\t"
	 "addc.u32         %3, %3, tmp;\n\t"

	 "mul.hi.u32       tmp, %5, %8;\n\t"
	 "add.cc.u32       %2, %2, tmp;\n\t"
	 "mul.hi.u32       tmp, %5, %9;\n\t"
	 "addc.u32         %3, %3, tmp;\n\t"

	 "mul.lo.u32       tmp, %6, %8;\n\t"
	 "add.cc.u32       %2, %2, tmp;\n\t"
	 "mul.lo.u32       tmp, %6, %9;\n\t"
	 "addc.u32         %3, %3, tmp;\n\t"

	 "mad.hi.u32       %3, %6, %8, %3;\n\t"

	 "mad.lo.u32       %3, %7, %8, %3;\n\t"
	 "}\n\t"
	 : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()), "=r"(result.word3())
	 : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
	   "r"(b.word0()), "r"(b.word1()), "r"(b.word2()), "r"(b.word3())
	 );
    return result;
    */

    u64 resultHi, resultLo;
    u64 aHi = a.getPart1(), aLo = a.getPart0();
    u64 bHi = b.getPart1(), bLo = b.getPart0();

    asm (
	 "{\n\t"
	 ".reg .u64 tmp;\n\t"
	 "mul.lo.u64    %0, %2, %4;\n\t"
	 "mul.hi.u64    %1, %2, %4;\n\t"
	 "mul.lo.u64    tmp, %2, %5;\n\t"
	 "add.u64       %1, %1, tmp;\n\t"
	 "mul.lo.u64    tmp, %3, %4;\n\t"
	 "add.u64       %1, %1, tmp;\n\t"
	 "}\n\t"
	 : "=l"(resultLo), "=l"(resultHi)
	 : "l"(aLo), "l"(aHi),
	   "l"(bLo), "l"(bHi));

    return Int128(resultHi, resultLo);

  }
#endif

  // multiply by a 128 bit value
  HD Int128 operator * (Int128 x) {
    return mult128lo(*this, x);
  }


  // breaking this out into a separate function to make it easier to
  // change the implementation
  HD static void divMod(u64 dividend, unsigned divisor,
			u64 &quotient, unsigned &remainder) {
    quotient = dividend / divisor;
    remainder = dividend % divisor;
  }


  HD Int128 operator / (unsigned divisor) {
    Int128 quotient;
    unsigned modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  HD unsigned operator % (unsigned divisor) {
    Int128 quotient;
    unsigned modulo;
    divMod(divisor, quotient, modulo);
    return modulo;
  }


  /*
    Divide a Int128 by a 32-bit value.
    If each letter is 32 bits:
      ABCD / E
  */
  HD void divMod(unsigned divisor, Int128 &quotient, unsigned &modulo) {
    Int128 thisCopy = *this;  // make a copy, in case 'quotient' is 'this'.
    u64 tmp, q;
    unsigned carry = 0;
    
    tmp = thisCopy.word3();
    divMod(tmp, divisor, q, carry);
    quotient.setWord3(q);

    tmp = ((u64)carry << 32) | thisCopy.word2();
    divMod(tmp, divisor, q, carry);
    quotient.setWord2(q);

    tmp = ((u64)carry << 32) | thisCopy.word1();
    divMod(tmp, divisor, q, carry);
    quotient.setWord1(q);

    tmp = ((u64)carry << 32) | thisCopy.word0();
    divMod(tmp, divisor, q, modulo);
    quotient.setWord0(q);
  }


  HD Int128 operator / (u64 divisor) {
    Int128 quotient;
    u64 modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  HD Int128 operator % (u64 divisor) {
    Int128 quotient;
    u64 modulo;
    divMod(divisor, quotient, modulo);
    return modulo;
  }

  HD void divMod(u64 divisor, Int128 &quotient, u64 &modulo) {
    if (divisor == 0) {
      modulo = 0xffffffffffffffffull;
      quotient.setPart0(0xffffffffffffffffull);
      quotient.setPart1(0xffffffffffffffffull);
      return;
    }

    if (divisor == 1) {
      modulo = 0;
      quotient = *this;
      return;
    }

    // if the divisor is 32 bits, use the simpler version
    if ((divisor & HI_MASK) == 0) {
      unsigned tmpMod;
      divMod((unsigned)divisor, quotient, tmpMod);
      modulo = tmpMod;
      return;
    }

    divideNewton(divisor, quotient, modulo);
    /*
    Int128 q;
    u64 m;
    divModKnuth(divisor, q, m);
    if (quotient != q || m != modulo) {
      printf("divMod error: %016llx%016llx / %llx\n  %016llx%016llx %llx\n  %016llx%016llx %llx\n", getPart1(), getPart0(), divisor, quotient.getPart1(), quotient.getPart0(), modulo, q.getPart1(), q.getPart0(), m);
    }
    */
  }

  /*
    Divide a Int128 by a 64-bit value.
    If each letter is 32 bits:
      ABCD / EF
  */
  HD void divModKnuth(u64 divisor, Int128 &quotient, u64 &modulo) {
    // the quotient will be placed here
    unsigned q[3];

    // shift divisor left so its top bit is set and split it into two
    // 32-bit values
    int divisorShift = XMP::countLeadingZeros64(divisor);
    unsigned d[2];
    d[0] = (unsigned) (divisor << divisorShift);
    d[1] = (unsigned) (divisor >> (32-divisorShift));

    // shift the dividend by the same amount plus 32 bits, and copy it into u[]
    // it is shifted by an extra 32 bits so needsCorrection() can
    // reference u[-1]
    unsigned uData[5], *u = uData+2;
    shiftAndCopy(uData, divisorShift);

    for (int j=2; j >= 0; j--,u--) {

      // one word of the quotient
      unsigned qWord;

      // qWord = u[j+1..j+2] / d[0..1]

      if (u[2] == d[1]) {
	qWord = UINT_MAX;
      } else {
	// estimate this word of the quotient by dividing by the top
	// word of the divisor
	qWord = (unsigned) (join64(u+1) / d[1]);

	// this would be needed if d[] had more than two elements
	// while (needsCorrection(qWord, u, d)) qWord--;
      }

      // remove qWord*d from u
      // if the qWord was too big, the result will be negative, so reduce
      // qWord and add back d until non-negative
      if (removeProduct(u, qWord, d)) {
	do {
	  qWord--;
	} while (!addBack(u, d));
      }
      q[j] = qWord;
    }

    quotient.setPart0(join64(q+0));
    quotient.setPart1(q[2]);

    // modulo is in uData[0..2] shifted by divisorShift
    modulo = (join64(uData) >> divisorShift) |
      ((u64)uData[2] << (64 - divisorShift));

    // test Newton method
    /*
    Int128 altQuotient;
    u64 altModulo;
    divideNewton(divisor, altQuotient, altModulo);
    if (altQuotient != quotient || altModulo != modulo) {
      printf("divide fail %016llx %016llx / %016llx\n", part1, getPart0(), divisor);
      assert(altQuotient == quotient);
      assert(altModulo == modulo);
    }
    */
  }

  
  // Use Newton-Raphson approximation to compute 2**128 / divisor
  HD static Int128 recipNewton(u64 divisor) {

    Int128 inva(0xffffffffffffffffuLL, -divisor);
    float frecip;
    Int128 x;

    int z = XMP::countLeadingZeros64(divisor);
    // 8388607 == 2**23 - 1
    // undershoot a bit (ha) to insure the estimate is low
    // ~2^(23+63-z) = ~2^(86-z) / divisor
    frecip = (8388607.0f * ((u64)1 << (63-z))) / divisor;
    // want 2^128 / divisor, have 2^(86-z) / divisor,
    // so shift by 128 - (86-z) = z + 42
    x = Int128(0, (u64)frecip) << (z+42);

    /*
    // 2**64 * (1-.5**22)
    // removing a small fraction to insure the initial value is low
    frecip = 1.8446739675663041e+19f / divisor;
    x.part1 = (u64)frecip;
    x.part0 = 0;
    */
      

    // Int128 inva = Int128(0) - divisor;
    // alternates tried, found slower:
    // Int128 inva(0xffffffffffffffffuLL, (~divisor)+1);

    // x = x + Int128::multiply_128_128_hi(x, inva*x);

    // start with reduced-precision operation, becuase only up to 46 bits
    // will be accurate anyway
    x.setPart1(x.getPart1() + XMP::multHi64(x.getPart1(), (inva.getPart0() * x.getPart1())));

    x = x + Int128::mult128hi(x, inva*x);
    x = x + Int128::mult128hi(x, inva*x);

    // final check
    u64 check = (Int128(0) - (x * divisor)).getPart0();
    if (check >= divisor) x += 1;

    return x;
  }


  /*
    Divide a Int128 by a 64-bit value using Newton-Raphson approximmation.

    1/a = 
    x(n+1) = x(n) * (2 - a * x(n))
           = 2*x(n) - a * x(n) * x(n)
	   = x(n) + x(n) - a * x(n) * x(n)
	   = x(n) + x(n) * (1 - a * x(n))
	  ~= x(n) + x(n) * (- a * x(n))
  */
  HD void divideNewton(u64 divisor, Int128 &quotient, u64 &modulo) {

    Int128 x = recipNewton(divisor);

    quotient = mult128hi(x, *this);
    modulo = (*this - quotient * divisor).getPart0();
    if (modulo >= divisor) {
      modulo -= divisor;
      quotient += 1;
    }
  }

  HD bool isZero() {
    return getPart1() == 0 && getPart0() == 0;
  }

  static unsigned rand32() {
    // rand only generates 31 bits
    return (rand() << 16) ^ rand();
  }

  static u64 rand64() {
    return ((u64)rand32() << 32) | rand32();
  }

  HD static void clearWords(u64 buf[], int len) {
    for (int i=0; i < len; i++) buf[i] = 0;
  }
  
  HD static void printWords(const char *label, u64 buf[], int len) {
#ifndef __CUDA_ARCH__
    cout << label << hex << setfill('0');
    for (int i=len-1; i >= 0; i--)
      cout << " " << setw(16) << buf[i];
    cout << endl;
#endif
  }

private:

  
  // shift this left by 'shift' bits (less than 32) and copy the
  // result into 192 bit array 'u' (u[0] is least significant), 
  // leaving u[0] empty (makes edge case easier later on)
  //   part1---- getPart0()----     0
  // u[5] u[4] u[3] u[2] u[1] u[0]
  HD void shiftAndCopy(unsigned u[], int shift) {
    u[0] = (unsigned) (getPart0() << shift);
    u[1] = (unsigned) (getPart0() >> (32-shift));
    u[2] = (unsigned) ((getPart1() << shift) | (getPart0() >> (64-shift)));
    u[3] = (unsigned) (getPart1() >> (32-shift));
    u[4] = (unsigned) (getPart1() >> (64-shift));
  }


  // given a pointer to two 32-bit integers, return them combined
  // into one 64-bit integers with the first one being less significant
  HD u64 join64(unsigned *p) {
    return p[0] | ((u64)p[1] << 32);
  }
   

  // removes qWord * (d1<<32 | d0) from u[0..2]
  // returns true iff result is negative
  HD bool removeProduct(unsigned *u, unsigned qWord, unsigned *d) {

    u64 pBig = (u64)d[0] * qWord;
    unsigned p = (unsigned) pBig;  // low 32 bits
    unsigned carry, nextCarry;

    carry = (u[0] < p) ? 1 : 0;
    u[0] -= p;
    pBig = (u64)d[1] * qWord + (pBig>>32);

    p = (unsigned) pBig;

    if (carry)
      nextCarry = u[1] <= p;
    else
      nextCarry = u[1] < p;

    // orig: u[1] - carry < p
    // fail: u[1] = 0, p = 1

    // attempt #2: nextCarry = (u[1] < p + carry) ? 1 : 0;
    // fail: u[1] = 0, p = 0xffffffff,c=1

    u[1] -= p + carry;
    carry = nextCarry;

    p = (unsigned) (pBig>>32);
    if (carry)
      nextCarry = u[2] <= p;
    else
      nextCarry = u[2] < p;
    u[2] -= p + carry;

    return nextCarry;
  }


  // returns true if qWord is too large:
  //   d[1]*qWord > (((u[1]<<32) + u[0] - qWord*d[0]) << 32) + u[-1]
  HD bool needsCorrection(unsigned qWord, unsigned *u, unsigned *d) {
    // copy qWord into a larger variable just to avoid ugly casts
    u64 qWordBig = qWord;

    u64 x = join64(u) - qWordBig * d[0];

    // if there are bit set in the high word, then when we shift
    // it up, it will definitely be greater than d[1]*qWord;
    if (x & HI_MASK) return true;
    
    return qWordBig * d[1] > (x<<32) + u[-1];
  }

  
  // returns 1 on overflow
  HD unsigned addBack(unsigned *u, unsigned *d) {
    // using extra long 'sum' just for the carry bit
    u64 sum = (u64)u[0] + d[0];
    u[0] = (unsigned) sum;
    sum = (u64)u[1] + d[1] + (sum>>32);
    u[1] = (unsigned) sum;
    sum = (u64)u[2] + (sum>>32);
    u[2] = (unsigned) sum;
    return (unsigned)(sum>>32);
  }
};


#ifdef __CUDACC__
class Int128Interleaved {
 public:
  unsigned words[BLOCK_SIZE*4];
  
  __device__ unsigned& operator [] (unsigned offset) {
    return words[offset*BLOCK_SIZE + threadIdx.x];
  }

  __device__ void set128(Int128 x) {
    (*this)[0] = x.word0();
    (*this)[1] = x.word1();
    (*this)[2] = x.word2();
    (*this)[3] = x.word3();
  }

  __device__ Int128 get128() {
    Int128 x((*this)[3], (*this)[2], (*this)[1], (*this)[0]);
    return x;
  }

  __device__ void multiply(Int128Interleaved &a, Int128Interleaved &b) {
    int carry;
    // row 0, lo words
    (*this)[0] = a[0] * b[0];
    (*this)[1] = a[0] * b[1];
    (*this)[2] = a[0] * b[2];
    (*this)[3] = a[0] * b[3];

    // row 0, hi words
    (*this)[1] = XMP::multHiAddCarryOut  (a[0], b[0], (*this)[1], carry);
    (*this)[2] = XMP::multHiAddCarryInOut(a[0], b[1], (*this)[2], carry);
    (*this)[3] = XMP::multHiAddCarryIn   (a[0], b[2], (*this)[3], carry);

    // row 1 lo
    (*this)[1] = XMP::multAddCarryOut  (a[1], b[0], (*this)[1], carry);
    (*this)[2] = XMP::multAddCarryInOut(a[1], b[1], (*this)[2], carry);
    (*this)[3] = XMP::multAddCarryIn   (a[1], b[2], (*this)[3], carry);

    // row 1 hi
    (*this)[2] = XMP::multHiAddCarryOut(a[1], b[0], (*this)[2], carry);
    (*this)[3] = XMP::multHiAddCarryIn (a[1], b[1], (*this)[3], carry);

    // row 2 lo
    (*this)[2] = XMP::multAddCarryOut  (a[2], b[0], (*this)[2], carry);
    (*this)[3] = XMP::multAddCarryInOut(a[2], b[1], (*this)[3], carry);

    // row 2 hi
    (*this)[3] = XMP::multHiAdd(a[2], b[0], (*this)[3]);

    // row 3 lo
    (*this)[3] = XMP::multAdd(a[3], b[0], (*this)[3]);
  }

  __device__ void copy(Int128Interleaved &a) {
    (*this)[0] = a[0];
    (*this)[1] = a[1];
    (*this)[2] = a[2];
    (*this)[3] = a[3];
  }
};
#endif


#endif // __INT_128__

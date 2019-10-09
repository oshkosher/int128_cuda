#ifndef __INT128TEST_H__
#define __INT128TEST_H__

#include <stdio.h>
#include "cudafree.h"
#include "int128.h"
#include "fixed_int.h"
#include "myassert.h"

class Int128Test {
 public:

  HD static void testConstructors() {
    printf("constructors\n");

    Int128 n1;
    // assertTrue(n1.getPart0() == 0);
    // assertTrue(n1.getPart1() == 0);

    n1 = 987654321;
    assertTrue(n1.getPart0() == 0x000000003ade68b1LL);
    assertTrue(n1.getPart1() == 0);

    n1 = 1000LL*1000*1000*1000*1000;
    assertTrue(n1.getPart0() == 0x00038d7ea4c68000LL);
    assertTrue(n1.getPart1() == 0);

    n1 = (u64)-1;
    assertTrue(n1.getPart0() == 0xffffffffffffffffLL);
    assertTrue(n1.getPart1() == 0);
    
    n1.setPart0(0xdeadbeef00112233LL);
    n1.setPart1(0xcafebabe55667788LL);
    Int128 n2(n1);
    assertTrue(n2.getPart0() == 0xdeadbeef00112233LL);
    assertTrue(n2.getPart1() == 0xcafebabe55667788LL);
    n1.setPart0(10);
    n1.setPart1(42);
    n2 = n1;
    assertTrue(n2.getPart0() == 10);
    assertTrue(n2.getPart1() == 42);
  }


  HD static void testWordAccessors() {
    Int128 n1, n2;
    printf("word accessors\n");

    n1.setPart0(0xdeadbeef00112233LL);
    n1.setPart1(0xcafebabe55667788LL);
    assertTrue(n1.word0() == 0x00112233);
    assertTrue(n1.word1() == 0xdeadbeef);
    assertTrue(n1.word2() == 0x55667788);
    assertTrue(n1.word3() == 0xcafebabe);

    n1.setWord0(10);
    n1.setWord1(20);
    n1.setWord2(30);
    n1.setWord3(40);
    assertTrue(n1.getPart0() == 0x000000140000000aLL);
    assertTrue(n1.getPart1() == 0x000000280000001eLL);
  }


  HD static void testAddWithCarry() {
    printf("addWithCarry\n");

    u64 x = 0xf001000200030004LL;
    u64 a = 0x0005000700090003LL;
    int carry = 0;
    x = XMP::addCarryInOut64(x, a, carry);
    assertTrue(carry == 0);
    assertTrue(x == 0xf0060009000c0007LL);
    assertTrue(a == 0x0005000700090003LL);

    carry = 1;
    x = XMP::addCarryInOut64(x, a, carry);
    assertTrue(carry == 0);
    assertTrue(x == 0xf00b00100015000bLL);
    
    carry = 1;
    x = XMP::addCarryInOut64(x, x, carry);
    assertTrue(carry == 1);
    assertTrue(x == 0xe0160020002a0017LL);

    x = XMP::addCarryOut64(x, a, carry);
    assertTrue(carry == 0);
    assertTrue(x == 0xe01b00270033001aLL);

    x = XMP::addCarryOut64(x, x, carry);
    assertTrue(carry == 1);
    assertTrue(x == 0xc036004e00660034LL);
  }

  HD static void testMultiply32() {
    Int128 n1, n2;

    printf("multiply by 32 bit value\n");

    n1.setPart0(0x0102030405060708LL);
    n1.setPart1(0xAABBCCDDEEFF0000LL);

    n2 = n1 * 2;
    assertTrue(n2.getPart0() == 0x020406080a0c0e10LL);
    assertTrue(n2.getPart1() == 0x557799bbddfe0000LL);
    
    n2 = n1 * 1000000000;
    assertTrue(n2.getPart0() == 0x53f49535d45c5000LL);
    assertTrue(n2.getPart1() == 0x933da8fb363c12b3LL);

    n2 = n1 * 1094835;
    assertTrue(n2.getPart0() == 0x4ec73fb830a88a98LL);
    assertTrue(n2.getPart1() == 0x3b3b3b29684d10d6LL);
  }

  HD static void testMultiply128() {
    Int128 n1, n2, n3;

    printf("multiply by 128 bit value\n");

    n1.setWord0(0x00000002);
    n1.setWord1(0x00000003);
    n1.setWord2(0x00000004);
    n1.setWord3(0x00000005);

    n2.setWord0(0x00000006);
    n2.setWord1(0x00000007);
    n2.setWord2(0x00000008);
    n2.setWord3(0x00000009);

    n3 = n1 * n2;
    assertTrue(n3.word0() == 12);
    assertTrue(n3.word1() == 2*7 + 3*6);
    assertTrue(n3.word2() == 2*8 + 3*7 + 4*6);
    assertTrue(n3.word3() == 2*9 + 3*8 + 4*7 + 5*6);

    n1 = 0xf0000000;
    n2 = 0xa0000000;
    n3 = n1 * n2;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0);
    n3 = n2 * n1;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0);

    n1.setWord1(0xc0000000);
    n3 = n1 * n2;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0x78000000);
    n3 = n2 * n1;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0x78000000);
    
    n1.setWord1(0);
    n2.setWord1(0xd0000000);
    n3 = n1 * n2;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0xc3000000);
    n3 = n2 * n1;
    assertTrue(n3.getPart0() == 0x9600000000000000LL);
    assertTrue(n3.getPart1() == 0xc3000000);

    n1 = 0xb0000000c0000000LL;
    n2 = 0x8000000090000000LL;
    n3 = n1 * n2;
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    assertTrue(n3.getPart1() == 0x58000000c3000000LL);
    n3 = n2 * n1;
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    assertTrue(n3.getPart1() == 0x58000000c3000000LL);

    n1.setPart1(0xa0000000);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xb2000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    n3 = n2 * n1;
    assertTrue(n3.getPart1() == 0xb2000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);

    n2.setPart1(0xF0000000);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    n3 = n2 * n1;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);

    n1.setPart1(0xe0000000a0000000LL);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    n3 = n2 * n1;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);

    n2.setPart1(0xd0000000f0000000LL);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);
    n3 = n2 * n1;
    assertTrue(n3.getPart1() == 0x66000000c3000000LL);
    assertTrue(n3.getPart0() == 0x6c00000000000000LL);

    n1 = 0xffffffff;
    n2 = 0xffffffff;
    n3 = n1 * n2;
    assertTrue(n3.getPart0() == 0xfffffffe00000001LL);
    assertTrue(n3.getPart1() == 0);

    n1 = 0xffffffffffffffffLL;
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xfffffffe);
    assertTrue(n3.getPart0() == 0xffffffff00000001LL);

    n2 = 0xffffffffffffffffLL;
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xfffffffffffffffeLL);
    assertTrue(n3.getPart0() == 1);

    n1.setPart1(0xffffffff);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xfffffffeffffffffLL);
    assertTrue(n3.getPart0() == 1);

    n2.setPart1(0xffffffff);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xfffffffe00000000LL);
    assertTrue(n3.getPart0() == 1);

    n1.setPart1(0xffffffffffffffffLL);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0xffffffff00000000LL);
    assertTrue(n3.getPart0() == 1);

    n2.setPart1(0xffffffffffffffffLL);
    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0);
    assertTrue(n3.getPart0() == 1);

    n1.setPart0(0x8472758392759376LL);
    n1.setPart1(0xbabcbdbebabfbdbcLL);
    n2.setPart0(0xbfbcbdbef832974bLL);
    n2.setPart1(0x9827394729348234LL);

    n3 = n1 * n2;
    assertTrue(n3.getPart1() == 0x4f8d690b118e0aa9LL);
    assertTrue(n3.getPart0() == 0x90371a545f78cd92LL);

    n1.setPart0(0xffffffffffffffffLL);
    n1.setPart1(0xffffffffffffffffLL);
    n2 = n1;
    n3 = n1 * n2;

    // ffffffffffffffff fffffffffffffffe 0000000000000000 0000000000000001
    assertTrue(n3.getPart1() == 0);
    assertTrue(n3.getPart0() == 1);


    n1 = Int128(0xffffffa, 0x00000023ffffff28ull);
    n3 = Int128(0xffffffa5fffffffull, 0xfffffffffffffaf0ull);
    u64 a = 0x100000006ull;
    n2 = n1 * a;
    assertTrue(n2 == n3);
  
    // n2 = n2 + 0x510;
    //   assertTrue(n2 == Int128(0xffffffa60000000ull, 0));

    n1 = Int128::mult64(0xfffffe0000000000ull, 0x20000000000ull);
    assertEqual64x(n1.getPart1(), 0x1fffffc0000ull);
    assertEqual64x(n1.getPart0(), 0);
  }


  HD static void testDivide32() {
    Int128 n1, n2;
    u64 quotient64;
    unsigned remainder32;
    printf("Int128.divMod(32)\n");

    n1 = Int128(0x2d50000013dLL, 0x24d00000133LL);
    n1.divMod(100, n2, remainder32);
    assertTrue(n2.getPart1() == 0x740000003LL);
    assertTrue(n2.getPart0() == 0x2b851ebe35c28f5fLL);

    Int128::divMod(1000, 17, quotient64, remainder32);
    assertTrue(quotient64 == 58);
    assertTrue(remainder32 == 14);

    Int128::divMod(0x9876543210fedcbaLL, 9823498, quotient64, remainder32);
    assertTrue(quotient64 == 1118345106297LL);
    assertTrue(remainder32 == 8772864);
  }


  HD static void testDivide64() {
    Int128 n1, n2, n3;

    printf("Int128.divMod(64)\n");

    n1 = 1000;
    u64 n5 = 195, n6;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 5);
    assertTrue(n3.getPart1() == 0);
    assertTrue(n6 == 25);

    n1 = 1000LL*1000*1000*1000 + 15;
    n5 = 1000LL*1000*1000*1000;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 1);
    assertTrue(n3.getPart1() == 0);
    assertTrue(n6 == 15);
    
    n1 = 0x80000002ULL << 32;
    n5 = (0x80000002ULL << 32) - 1;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 1);
    assertTrue(n3.getPart1() == 0);
    assertTrue(n6 == 1);

    n1 = Int128(0x361e7574ee369000LL, 0x5bd17ec67cbe1800LL);
    n5 = 0x54ea70ff4efc3000LL;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 0xa327c052610762d0LL);
    assertTrue(n3.getPart1() == 0);
    assertTrue(n6 == 0x963dc0475771800LL);

    n1 = Int128(0x361e7574ee369000LL, 0x5bd17ec67cbe1800LL);
    n5 = 0xbeefLL;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 0xf3db361d9b213990LL);
    assertTrue(n3.getPart1() == 0x488fc90dc7e0LL);
    assertTrue(n6 == 0x7a90);

    n1 = Int128(0x0ffffffa60000000ull, 0);
    n5 = 0x0000000100000006ull;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 0x00000023ffffff28ull);
    assertTrue(n3.getPart1() == 0xffffffa);
    assertTrue(n6 == 0x510);

    n1 = Int128(0x1000000000ull, 0);
    n5 = 0x100000045ull;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 0xfffffbb00001298full);
    assertTrue(n3.getPart1() == 0xf);
    assertTrue(n6 == 0xffafcc75);

    n1 = Int128(0x0fffffff10000000ull, 0);
    n5 = 0x100000001ull;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart0() == 0x00000000ffffffffull);
    assertTrue(n3.getPart1() == 0xfffffff);
    assertTrue(n6 == 0x1);

    n1 = Int128(0, 0x0000000c24cd0369ull);
    n1 = n1 * n1;
    assertTrue(n1.getPart0() == 0x78829efbfe35a111ull);
    assertTrue(n1.getPart1() == 0x93);
    n1.divMod(79992000121ull, n3, n6);
    assertTrue(n3.getPart1() == 0);
    assertTrue(n3.getPart0() == 0x7eb0670dcull);
    assertTrue(n6 == 0xff5324115ull);

    n1 = Int128(0x8000000000000000ull, 0);
    n5 = 1;
    n1.divMod(n5, n3, n6);
    assertTrue(n3.getPart1() == 0x8000000000000000ull);
    assertTrue(n3.getPart0() == 0);
    assertTrue(n6 == 0);
  }


  static void testRandomizedMultDiv() {
    printf("randomized mult / div\n");

#ifndef __CUDACC__
    srand(0xf0f0f0f0);
#endif
  
    for (int i=0; i < 100; i++) {
      u64 x = Int128::rand32();
      u64 y = Int128::rand32();
      u64 z = x * y;

      for (int xShift = 0; xShift < 128; xShift++) {
	Int128 x1 = Int128(x) << xShift;
	for (int yShift = 0; yShift < (128-xShift); yShift++) {
	  Int128 y1 = Int128(y) << yShift;
	  int shift = xShift + yShift;

	  Int128 prod = x1 * y1;
	  if (!(Int128(z) << shift == prod)) {
	    assert (0);
	    /*
	      cout << "Failed (" << x << "<<" << xShift << ")*(" << y << 
	      "<<" << yShift << "), expected (" << z << "<<" << shift <<
	      "), got " << prod << endl;
	    */
	  }
	  assertTrue(Int128(z) << shift == prod);

	  u64 zMissingBits = z;
	  if (shift > 64) {
	    zMissingBits = (zMissingBits << (shift-64)) >> (shift-64);
	  }
	  if (!(prod >> shift == zMissingBits)) {
	    assertTrue(0);
	    /*
	      cout << "Failed " << x << "<<" << xShift << "*" << y << "<<" <<
	      yShift << endl;
	    */
	  }

	  assertTrue(prod >> shift == zMissingBits);

	  Int128 quotient;
	  u64 modulo;

	  if (shift <= 64) {
	    prod.divMod(x, quotient, modulo);
	    assertTrue(modulo == 0);
	    assertTrue(quotient >> shift == y);
	    prod.divMod(y, quotient, modulo);
	    assertTrue(modulo == 0);
	    assertTrue(quotient >> shift == x);
	  }
	}
      }
    }
  }


  HD static void testDivideNewton() {
    // 300000000000000000000000000000000000017
    Int128 x(0xe1b1e5f90f944d6eull, 0x1c9e66c000000011ull);
    Int128 quotient;
    u64 modulo;
    x.divideNewton(30, quotient, modulo);
    assertTrue(quotient.getPart0() == 0x00f436a000000000ull);
    assertTrue(quotient.getPart1() == 0x0785ee10d5da46d9ull);
    assertTrue(modulo == 17);

    x = Int128(0, 0x097d91c74dba5fc0ull);
    x.divideNewton(0x1000f8401ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0);
    assertTrue(quotient.getPart0() == 0x97cfe8f);
    assertTrue(modulo == 0xe69aa531);

    x = Int128(0x012d432f10000000ull, 0);
    x.divideNewton(0x1000f8401ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0x12d30ed);
    assertTrue(quotient.getPart0() == 0xe3a7ceda76e29491ull);
    assertTrue(modulo == 0x10003a76full);

    x = Int128(0, 0x097d91c74dba5fc0ull);
    x.divMod(0x1000f8401ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0);
    assertTrue(quotient.getPart0() == 0x97cfe8f);
    assertTrue(modulo == 0xe69aa531);

    x = Int128(0x012d432f10000000ull, 0);
    x.divMod(0x1000f8401ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0x12d30ed);
    assertTrue(quotient.getPart0() == 0xe3a7ceda76e29491ull);
    assertTrue(modulo == 0x10003a76full);

    x = Int128(0x066c102d60000000ull, 0);
    x.divMod(0x1007af2b9ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0x668fc11);
    assertTrue(quotient.getPart0() == 0x5f91df957cfc8c77ull);
    assertTrue(modulo == 1);

    x = Int128(0x066c102d60000000ull, 0);
    x.divideNewton(0x1007af2b9ull, quotient, modulo);
    assertTrue(quotient.getPart1() == 0x668fc11);
    assertTrue(quotient.getPart0() == 0x5f91df957cfc8c77ull);
    assertTrue(modulo == 1);
  }


  HD static void test64hi() {
    printf("test64hi\n");

    Int128 x(0xFEDCBA9876543210ll, 0x13243546576879ABll);
    Int128 y(0xfeedbabedeadcafell, 0x9874837592837492ll);

    u64 h;
    h = XMP::multHi64(x.getPart0(), y.getPart0());
    assertEqual64x(h, 0x0b6635de39897615ll);
    h = XMP::multHi64(x.getPart0(), y.getPart1());
    assertEqual64x(h, 0x130fb357b1139effll);

    Int128 lo = x*y;
    Int128 hi = Int128::mult128hi(x, y);

    // fdcbad6669a2c7d1 3c65f3da507f9466 d83aec2c46a5e8df c620adb9f937df86
    assertTrue(lo.getPart0() == 0xc620adb9f937df86ll);
    assertTrue(lo.getPart1() == 0xd83aec2c46a5e8dfll);
    assertTrue(hi.getPart0() == 0x3c65f3da507f9466ll);
    assertTrue(hi.getPart1() == 0xfdcbad6669a2c7d1ll);

    x = Int128(0xffffffffffffffffLL, 0xffffffffffffffffLL);
    y = Int128(0xffffffffffffffffLL, 0xffffffffffffffffLL);
    lo = x*y;
    hi = Int128::mult128hi(x, y);
    assertTrue(lo.getPart0() == 1);
    assertTrue(lo.getPart1() == 0);
    printf("hi = %016llx %016llx\n", hi.getPart1(), hi.getPart0());
    assertTrue(hi.getPart0() == 0xfffffffffffffffeLL);
    assertTrue(hi.getPart1() == 0xffffffffffffffffLL);
  }

  HD static void testUNR() {
    printf("testUNR\n");

    /*
      for (u64 a = 1; a < 1000000; a++) {
      Int128 expected = Int128::fract(1, a);
      float frecip = (float)(1<<23) / a;
      Int128 x((u64)frecip << 41, 0);
      if (x > expected) {
      printf("a = %llu\n", a);
      printf("exp %016llx%016llx\n", expected.getPart1(), expected.part0);
      printf("  x %016llx%016llx\n", x.getPart1(), x.part0);
      }
      }
    */

    u64 a = 1234123423423LLU;
    //      18446739675663040512
    Int128 expected = FixedIntUtil::fract128(1, a);
    printf("    %016llx%016llx\n", expected.getPart1(), expected.getPart0());
    float frecip = 18446739675663040512.0f / a;  // 2**64 * (1-.5**22)
    printf("frecip = %.0f\n", frecip);
    Int128 x((u64)frecip, 0);
    Int128 inva = Int128(0) - a;
    Int128 add;

    for (int i=0; i < 10; i++) {
      printf("x = %016llx%016llx\n", x.getPart1(), x.getPart0());
    
      add = Int128::mult128hi(x, inva*x);
      /*
	if (add.getPart1()) {
	printf("  add = %llx%016llx\n", add.getPart1(), add.getPart0());
	} else {
	printf("  add = %llx\n", add.getPart0());
	}
      */
      if (add == 0) break;
      x = x + add;
    }

    // final check
    if (Int128(0) - (x * a) >= a) x += 1;
    printf("    %016llx%016llx\n", x.getPart1(), x.getPart0());
  }


  HD static int test() {
    testConstructors();
    testWordAccessors();
    testAddWithCarry();
    testMultiply32();
    testMultiply128();
    testDivide32();
    testDivide64();
#ifndef __CUDACC__
    testRandomizedMultDiv();
#endif
    testDivideNewton();
    test64hi();
    testUNR();

    printf("test complete.\n");
    return 0;
  }
};  

#endif // __INT128TEST_H__

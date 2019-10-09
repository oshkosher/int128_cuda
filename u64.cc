#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "u64.h"


bool stringToU64(u64 &result, const char *str) {
  // skip leading whitespace
  while (isspace(*str)) str++;

  // make sure it starts with a digit, or since we did guarantee to ignore
  // commas, ignore a leading comma.
  if (!isdigit(*str) && (*str != ',')) return false;
  result = 0;
  u64 pos = 1;
  while (isdigit(*str) || (*str == ',')) {
    if (isdigit(*str))
      result = result * 10 + pos * (*str - '0');
    str++;
  }
  return true;
}


// max value: 18446744073709551615
std::string u64ToString(u64 x) {
  char buf[21];
  sprintf(buf, "%llu", x);
  return std::string(buf);
}

// max value: 18,446,744,073,709,551,615
std::string commafy(u64 x) {
  char buf[26], *p;
  if (x == 0) return "0";
  p = buf+25;
  int i = 2;
  while (x>0) {
    *p-- = (x % 10) + '0';
    x /= 10;
    if (i==0 && x > 0) {
      *p-- = ',';
      i = 2;
    } else {
      i--;
    }
  }
  return std::string(p+1, buf+25-p);
}


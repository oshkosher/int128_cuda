#!/usr/bin/python

import sys

def printHelp():
  print """
  write_mult.py <f1Pat> <f1Len> <f2Pat> <f2len> <resultPat> <resultLen> <resultOffset>
    f1Pat, f2Pat, and resultPat are python printf-style patterns
    into which integers 0..len-1 will be substituted.

    ./write_mult.py a[%d] 4 b[%d] 4 result[%d] 8 0
"""
  sys.exit(0)


class Array:
  def __init__(self, pattern, len, offset=0):
    self.pattern = pattern
    self.len = len
    self.offset = offset

  def __getitem__(self, index):
    if index < self.offset:
      return "dummy%d" % index
    else:
      return self.pattern % (index-self.offset)
  

def main(args):
  if len(args) != 7: printHelp()

  (f1Pat, f1Len, f2Pat, f2Len, resultPat, resultLen, resultOffset) = args
  f1Len = int(f1Len)
  f2Len = int(f2Len)
  resultLen = int(resultLen)
  resultOffset = int(resultOffset)

  f1 = Array(f1Pat, f1Len)
  f2 = Array(f2Pat, f2Len)
  result = Array(resultPat, resultLen, resultOffset)

  if f1Len > f2Len:
    write_mult(f2, f1, result, resultOffset)
  else:
    write_mult(f1, f2, result, resultOffset)


def write_mult(f1, f2, result, resultOffset):
  # f1.len <= f2.len
  len = result.len + resultOffset

  print '{'
  if len > 1:
    print 'int carry;'
  if f2.len+1 < len:
    print 'unsigned temp;'
  if resultOffset > 0:
    dummyList = map(lambda i: 'dummy%d' % (i+1), range(resultOffset-1))
    print 'unsigned %s;' % (', '.join(dummyList))

  # do the first row of low words
  firstRowLow = 0
  if resultOffset > 0:
    firstRowLow = 1
  for i in range(firstRowLow, min(f2.len, len)):
    print '%s = %s * %s;' % (result[i], f1[0], f2[i])

  # do the first row of high words
  if len > 1:
    print '%s = XMP::multHiAddCarryOut(%s, %s, %s, carry);' % (
      result[1], f1[0], f2[0], result[1])
  for i in range(1, min(f2.len-1, len-1)):
    print '%s = XMP::multHiAddCarryInOut(%s, %s, %s, carry);' % (
      result[i+1], f1[0], f2[i], result[i+1])
  if f2.len < len:
    print '%s = XMP::multHiAddCarryIn(%s, %s, 0, carry);' % (
      result[f2.len], f1[0], f2[f2.len-1])

  # remaining words  
  for i1 in range(1, min(f1.len, len)):
    print
    # low words
    print '%s = XMP::multAddCarryOut(%s, %s, %s, carry);' % (
      result[i1], f1[i1], f2[0], result[i1])
    
    for i2 in range(1, min(f2.len-1, len-i1)):
      print '%s = XMP::multAddCarryInOut(%s, %s, %s, carry);' % (
        result[i1+i2], f1[i1], f2[i2], result[i1+i2])
    if i1+f2.len-1 < len:
      print '%s = XMP::multAddCarryInOut(%s, %s, %s, carry);' % (
        result[i1+f2.len-1], f1[i1], f2[f2.len-1], result[i1+f2.len-1])
    if i1+f2.len < len:
      print 'temp = XMP::addCarryIn(0, 0, carry);'
    print

    # hi words
    if i1+1 < len:
      print '%s = XMP::multHiAddCarryOut(%s, %s, %s, carry);' % (
        result[i1+1], f1[i1], f2[0], result[i1+1])
    
    for i2 in range(1, min(f2.len-1, len-i1-1)):
      print '%s = XMP::multHiAddCarryInOut(%s, %s, %s, carry);' % (
        result[i1+i2+1], f1[i1], f2[i2], result[i1+i2+1])
    if i1+f2.len < len:
      print '%s = XMP::multHiAddCarryIn(%s, %s, temp, carry);' % (
        result[i1+f2.len], f1[i1], f2[f2.len-1])

  print '}'



if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

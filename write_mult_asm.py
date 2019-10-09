#!/usr/bin/python

import sys

result = None
a = None
b = None
madc_supported = 0

def printHelp():
  print """
  write_mult_asm.py <ptxver> <len> <part> <f1Pat> <f2Pat> <resultPat>
    f1Pat, f2Pat, and resultPat: python printf-style patterns
      if resultPat has two comma-separated parts and part==full, then the
      first is used to store the high half of the result and the
      second the low half:  <resultHiPat,resultLoPat>
    ptxver: 2 or 3, because PTX version 3 added multiply-add instructions.
      PTX 3 is supported by CUDA toolkits 4.2 and later.
    len: number of 32-bit words in the factors
    part: lo, hi, or full, for the size of the result

  write_mult_asm.py 3 4 full 'a.word%d()' 'b.word%d()' 'lo.word%d(),hi.word%d()'
"""
  sys.exit(0)

class Array:
  def __init__(self, pattern, length, asm_offset, is_output_param):
    self.pattern = pattern
    self.length = length
    self.asm_offset = asm_offset
    self.is_output_param = is_output_param

  def __getitem__(self, index):
    return '%%%d' % (self.asm_offset + index)

  def asmParam(self, index):
    if self.is_output_param:
      prefix = '"=r"'
    else:
      prefix = '"r"'
    return prefix+'('+(self.pattern % index)+')'

  def getParams(self):
    return ', '.join(map(lambda i: self.asmParam(i), range(self.length)))

class ArrayPair:
  def __init__(self, pairPattern, halfLength, asm_offset, is_output_param):
    self.halfLength = halfLength
    self.asm_offset = asm_offset
    self.is_output_param = is_output_param
    patterns = pairPattern.split(',')
    if len(patterns) != 2:
      print 'Pair pattern ("fooLo,fooHi") required: ' + pairPattern
      sys.exit(1)
    (self.patternLo, self.patternHi) = patterns

  def __getitem__(self, index):
    return '%%%d' % (self.asm_offset + index)

  def asmParam(self, index):
    if self.is_output_param:
      prefix = '"=r"'
    else:
      prefix = '"r"'

    if index < self.halfLength:
      name = self.patternLo % index
    else:
      name = self.patternHi % (index - self.halfLength)

    return prefix+'('+name+')'

  def getParams(self):
    return ', '.join(map(lambda i: self.asmParam(i), range(self.halfLength*2)))

class ArrayHi:
  def __init__(self, pattern, tmpPattern, length, asm_offset, is_output_param):
    self.pattern = pattern
    self.tmpPattern = tmpPattern
    self.length = length
    self.asm_offset = asm_offset
    self.is_output_param = is_output_param

  def __getitem__(self, index):
    if index < self.length:
      return self.tmpPattern % index
    else:
      return '%%%d' % (self.asm_offset + index - self.length)

  def asmParam(self, index):
    if self.is_output_param:
      prefix = '"=r"'
    else:
      prefix = '"r"'
    return prefix+'('+(self.pattern % index)+')'

  def getParams(self):
    return ', '.join(map(lambda i: self.asmParam(i), range(self.length)))

def replaceTab(s):
  i = s.find('\t')
  if i == -1:
    return s
  else:
    return s[:i] + ' ' + (' ' * (16-i)) + s[i+1:]

def writeLine(s):
  s = replaceTab(s)
  print '"' + s + ';\\n\\t"'

def mul(ai, bi):
  writeLine('mul.lo.u32\t%s, %s, %s' % (result[ai+bi], a[ai], b[bi]))

def mad(side, carry, ai, bi, addendOverride=None):
  global madc_supported, result, f1, f2

  if side != 'hi' and side != 'lo':
    print 'mad() invalid side: "%s"' % side
    sys.exit(1);
  if carry != '' and carry != 'in' and carry != 'inout' and carry != 'out':
    print 'mad() invalid carry: "%s"' % carry
    sys.exit(1);

  resulti = ai + bi
  if side == 'hi':
    resulti += 1

  if addendOverride==None:
    addend = result[resulti]
  else:
    addend = str(addendOverride)

  if madc_supported:
    if carry == 'out':
      instr = 'mad.%s.cc.u32' % side
    elif carry == 'inout':
      instr = 'madc.%s.cc.u32' % side
    elif carry == 'in':
      instr = 'madc.%s.u32' % side
    else:
      instr = 'mad.%s.u32' % side
    writeLine('%s\t%s, %s, %s, %s' %
              (instr, result[resulti], a[ai], b[bi], addend))
  else:
    if carry == '':
      writeLine('mad.%s.u32\t%s, %s, %s, %s' %
                (side, result[resulti], a[ai], b[bi], addend))
    else:
      writeLine('mul.%s.u32\ttmp, %s, %s' % (side, a[ai], b[bi]))
      if carry == 'out':
        instr = 'add.cc.u32'
      elif carry == 'inout':
        instr = 'addc.cc.u32'
      else:
        instr = 'addc.u32'
      writeLine('%s\t%s, %s, tmp' %
                (instr, result[resulti], addend))
  

def write_mult_asm_full(length):
  global result, f1, f2

  print 'asm( "{\\n\\t"'

  if not madc_supported:
    writeLine('.reg .u32 tmp')

  # do the first row of low words
  for i in range(length):
    mul(0, i)
  print

  # do the first row of high words
  mad('hi', 'out', 0, 0)
  for i in range(1, length-1):
    mad('hi', 'inout', 0, i)
  mad('hi', 'in', 0, length-1, '0')
  print

  for ai in range(1, length):
    # row of low words
    mad('lo', 'out', ai, 0)
    for bi in range(1, length-1):
      mad('lo', 'inout', ai, bi)
    mad('lo', 'in', ai, length-1)
    print

    # row of high words
    mad('hi', 'out', ai, 0)
    for bi in range(1, length-1):
      mad('hi', 'inout', ai, bi)
    mad('hi', 'in', ai, length-1, '0')
    print

  print '"}\\n\\t"'

  print ': ' + result.getParams()
  print ': ' + a.getParams() + ',\n  ' + b.getParams()
  print ');'


def write_mult_asm_lo(length):
  global result, f1, f2

  print 'asm( "{\\n\\t"'

  if not madc_supported:
    writeLine('.reg .u32 tmp')

  # do the first row of low words
  for i in range(length):
    mul(0, i)
  print

  # do the first row of high words
  mad('hi', 'out', 0, 0)
  for i in range(1, length-2):
    mad('hi', 'inout', 0, i)
  mad('hi', 'in', 0, length-2)
  print

  for ai in range(1, length-1):
    # row of low words
    mad('lo', 'out', ai, 0)
    for bi in range(1, length-1-ai):
      mad('lo', 'inout', ai, bi)
    mad('lo', 'in', ai, length-1-ai)
    print

    # row of high words
    if ai < length-2:
      mad('hi', 'out', ai, 0)
      for bi in range(1, length-2-ai):
        mad('hi', 'inout', ai, bi)
      mad('hi', 'in', ai, length-2-ai)
      print

  # last element
  mad('hi', '', length-2, 0)
  print
  mad('lo', '', length-1, 0)

  print '"}\\n\\t"'

  print ': ' + result.getParams()
  print ': ' + a.getParams() + ',\n  ' + b.getParams()
  print ');'


def write_mult_asm_hi(length):
  global result, f1, f2

  print 'asm( "{\\n\\t"'

  if not madc_supported:
    writeLine('.reg .u32 tmp')

  tmp_regs = ', '.join(map(lambda i: 'tmp%d' % i, range(1, length)))
  writeLine('.reg .u32 ' + tmp_regs)
  print

  # do the first row of low words
  for i in range(1, length):
    mul(0, i)
  print

  # do the first row of high words
  mad('hi', 'out', 0, 0)
  for i in range(1, length-1):
    mad('hi', 'inout', 0, i)
  mad('hi', 'in', 0, length-1, '0')
  print

  for ai in range(1, length):
    # row of low words
    mad('lo', 'out', ai, 0)
    for bi in range(1, length-1):
      mad('lo', 'inout', ai, bi)
    mad('lo', 'in', ai, length-1)
    print

    # row of high words
    mad('hi', 'out', ai, 0)
    for bi in range(1, length-1):
      mad('hi', 'inout', ai, bi)
    mad('hi', 'in', ai, length-1, '0')
    print

  print '"}\\n\\t"'

  print ': ' + result.getParams()
  print ': ' + a.getParams() + ',\n  ' + b.getParams()
  print ');'


def main(args):
  global result, a, b, madc_supported
  if len(args) != 6: printHelp()

  if args[0] == '2':
    madc_supported = 0
  elif args[0] == '3':
    madc_supported = 1
  else:
    print 'Unrecognized ptxver; must be 2 or 3.'
    return 1

  length = int(args[1])
  part = args[2]
  if part != 'lo' and part != 'hi' and part != 'full':
    print 'Invalid part: ' + part
    return 1

  if part == 'full':
    resultLen = length*2
  else:
    resultLen = length

  a = Array(args[3], length, resultLen, 0)
  b = Array(args[4], length, resultLen+length, 0)

  if args[5].find(',') == -1:
    if part == 'hi':
      result = ArrayHi(args[5], 'tmp%d', length, 0, 1)
    else:
      result = Array(args[5], resultLen, 0, 1)
  else:
    if part != 'full':
      print 'Split pattern (lo%%d,hi%%d) only allowed when part="full"'
      sys.exit(1)
    result = ArrayPair(args[5], length, 0, 1)

  if part == 'full':
    write_mult_asm_full(length)
  elif part == 'lo':
    write_mult_asm_lo(length)
  elif part == 'hi':
    write_mult_asm_hi(length)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

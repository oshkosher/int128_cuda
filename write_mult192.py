#!/usr/bin/python

import sys

madc_supported = 0

def replaceTab(s):
  i = s.find('\t')
  if i == -1:
    return s
  else:
    return s[:i] + ' ' + (' ' * (16-i)) + s[i+1:]

def writeLine(s):
  s = replaceTab(s)
  print '"' + s + ';\\n\\t"'

# replace first '@' character with integer i
def replaceIndex(s, i):
  offset = s.find('@')
  if offset >= 0:
    return s[:offset] + str(i) + s[offset+1:]
  else:
    return s

def mad(side, carry, ai, bi, addendOverride=None):
  global madc_supported

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
    addend = 'result%d' % resulti
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
    writeLine('%s\tresult%d, a%d, b%d, %s' %
              (instr, resulti, ai, bi, addend))
  else:
    if carry == '':
      writeLine('mad.%s\tresult%d, a%d, b%d, %s' %
                (side, resulti, ai, bi, addend))
    else:
      writeLine('mul.%s.u32\ttmp, a%d, b%d' % (side, ai, bi))
      if carry == 'out':
        instr = 'add.cc.u32'
      elif carry == 'inout':
        instr = 'addc.cc.u32'
      else:
        instr = 'addc.u32'
      writeLine('%s\tresult%d, %s, tmp' %
                (instr, resulti, addend))


def main(args):
  global madc_supported

  if len(args) != 5:
    print """
  write_mult192.py <fmt> <ptxver> <factor1> <factor2> <result>
  fmt:
    array32 - arguments are 32 bit array pointers
    array64 - arguments are 64 bit array pointers
    64bit - arguments are multiple 64-bit values aN, bN, resultN
  ptxver:
    2: does not support madc or mad.cc
    3: supports madc and mad.cc
  factor1, factor2, result
    Templates for variable where '@' character is replaced with index
    For example: a.getWord@(), result.refPart(@)
"""
    return 0

  fmt = args[0]
  if fmt != 'array32' and fmt != 'array64' and fmt != '64bit':
    print 'Invalid <fmt>'
    return 1
  ptxver = int(args[1])
  if ptxver != 2 and ptxver != 3:
    print 'Invalid <ptxver>'
    return 1

  if ptxver == 3:
    madc_supported = 1

  factor1 = args[2]
  factor2 = args[3]
  result = args[4]

  print 'asm ("{\\n\\t"'
  writeLine('.reg .u32 result<12>')
  writeLine('.reg .u32 a<6>')
  writeLine('.reg .u32 b<6>')
  if not madc_supported:
    writeLine('.reg .u32 tmp')

  if fmt == '64bit':
    for i in xrange(3):
      writeLine('mov.b64\t{a%d,a%d},%%%d' % (i*2, i*2+1, 6+i))
      writeLine('mov.b64\t{b%d,b%d},%%%d' % (i*2, i*2+1, 9+i))
                
  else:
    for i in xrange(6):
      writeLine('ld.b32\ta%d, [%%2+%d]' % (i, i*4))
      writeLine('ld.b32\tb%d, [%%3+%d]' % (i, i*4))
  
  # compute lo parts of row 1
  for i in xrange(6):
    writeLine('mul.lo.u32\tresult%d, a0, b%d' % (i, i))
  
  # compute hi parts of row 1
  mad('hi', 'out', 0, 0)
  for i in xrange(1, 5):
    mad('hi', 'inout', 0, i)
  mad('hi', 'in', 0, 5, 0)
  
  for i in xrange(1, 6):
    mad('lo', 'out', i, 0)
    for j in xrange(1, 6):
      mad('lo', 'inout', i, j)
    writeLine('addc.u32\tresult%d, 0, 0' % (i+6))

    mad('hi', 'out', i, 0)
    for j in xrange(1, 5):
      mad('hi', 'inout', i, j)
    mad('hi', 'in', i, 5)
  
  
  if fmt == '64bit':
    for i in xrange(6):
      writeLine('mov.b64\t%%%d,{result%d,result%d}' % (i, i*2, i*2+1))

  else:
    for i in xrange(6):
      writeLine('st.b32\t[%%0+%d], result%d' % (i*4, i))
    for i in xrange(6):
      writeLine('st.b32\t[%%1+%d], result%d' % (i*4, i+6))
  
  
  print '"}\\n"'

  if fmt == '64bit':
    print ':'
    for i in xrange(6):
      sys.stdout.write('"=l"(%s)' % replaceIndex(result, i))
      if i < 5:
        sys.stdout.write(', ')
    print '\n: '
    for i in xrange(3):
      sys.stdout.write('"l"(%s), ' % replaceIndex(factor1, i))
    for i in xrange(3):
      sys.stdout.write('"l"(%s)' % replaceIndex(factor2, i))
      if i < 2:
        sys.stdout.write(', ')
  elif fmt == 'array32':
    print '::'
    print '"r"(result_lo), "r"(result_hi), "r"(a), "r"(b)'
  else:
    print '::'
    print '"l"(result_lo), "l"(result_hi), "l"(a), "l"(b)'

  print ');'

if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))

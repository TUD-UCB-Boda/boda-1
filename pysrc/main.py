"""Winograd transformation code generator.

Usage:
    main.py gen <m> <r> [--unroll=<u>] [--disable-cache]
    main.py test <m> <r> <p>

Options:
    --unroll=<u>          Unroll factor [default: 1].
    -h --help           Show this screen.
    --version           Show version.
"""

import csv
import os

from tc_gen import TransformationCodeGenerator
from docopt import docopt
from schema import Schema, Use, SchemaError
from sympy import Rational, Integer, srepr

At2 = ''
At = ''
Bt2 = ''
Bt = ''
Gg = ''
Gg2 = ''

docopt_schema = Schema({
        '<m>': Use(int, error='Output tile size should be integer'),
        '<r>': Use(int, error='Kernel size should be integer'),
        '<p>': Use(str, error='Polynomials should be list'),
        '--unroll': Use(int, error='Unroll factor should be integer'),
        '--disable-cache': Use(bool, error='disable cache should be bool'),
        'gen': Use(bool, error='gen should be bool'),
        'test': Use(bool, error='test should be bool')
    })

def F(m, r, u, c=False):
    global At2
    global At
    global Bt2
    global Bt
    global Gg
    global Gg2
    print "arguments", m, r, u, c;
    tc_gen = TransformationCodeGenerator(m, r, c)
    pps = tc_gen.gen_points()
    mean_error, median_error = tc_gen.test_winograd(m, r, pps)

    if not os.path.exists('.winograd_pp.cache'):
        with open('.winograd_pp.cache', mode='w') as results_pp:
            results_pp = csv.writer(results_pp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_pp.writerow(['m', 'r', 'alpha', 'polynomials', 'mean_error', 'median_error'])
    
    with open('.winograd_pp.cache', mode='a') as results_pp:
        results_pp = csv.writer(results_pp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_pp.writerow([m, r, m+r-1, str([srepr(x) for x in pps]).replace("'",""), mean_error, median_error])
    if pps:
        tc_gen.gen_code(u)
        Gg = tc_gen.Gg
        Gg2 = tc_gen.Gg2
        Bt = tc_gen.Bt
        Bt2 = tc_gen.Bt2
        At = tc_gen.At
        At2 = tc_gen.At2
        print "after code gen:", At


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Winograd transformation code generator v0.1')
    try:
        arguments = docopt_schema.validate(arguments)
    except SchemaError as e:
        exit(e)

    m, r = arguments['<m>'], arguments['<r>']

    if arguments['test']:
        p = eval(arguments['<p>'])
        tc = TransformationCodeGenerator(m, r, 0)
        tc.test_winograd(m, r, p)
    elif arguments['gen']:
        F(m, r, arguments['--unroll'], arguments['--disable-cache'])
    else:
        print "Nothing to do!"

def getAt():
    return At
def getAt2():
    return At2
def getBt():
    return Bt
def getBt2():
    return Bt2
def getGg():
    return Gg 
def getGg2():
    return Gg2

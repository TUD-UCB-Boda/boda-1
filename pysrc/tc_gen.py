import sys
import os
import pandas
import csv
import numpy as np
import sympy
from sympy import Integer, Rational, srepr, Matrix
from sympy.solvers.solveset import linsolve
from scipy.signal import correlate2d

import time

import symbolic

class TransformationCodeGenerator(object):
    def __init__(self, out_size, filts_size, disable_cache):
        self.out_size = out_size
        self.filts_size = filts_size
        self.alpha = out_size + filts_size - 1
        self.At = list()
        self.Bt = list()
        self.G = list()
        self.fAt = list()
        self.fBt = list()
        self.fG = list()
        self.At2 = ''
        self.At = ''
        self.Bt2 = ''
        self.Bt = ''
        self.Gg = ''
        self.Gg2 = ''
        self.disable_cache = disable_cache

    def __M(self, x, list_frac_p):
        M1 = Rational(1, 1)
        for j in range(len(list_frac_p)):
            M1 = M1 * (x - list_frac_p[j])
        return M1

    def __Mi(self, x, i, list_frac_p):
        M1 = Rational(1, 1)
        for j in range(len(list_frac_p)):
            if j != i:
                M1 = M1 * (x - list_frac_p[j])
        return M1

    def __N(self, i, list_frac_p):
        M1 = Rational(1, 1)
        for j in range(len(list_frac_p)):
            if j != i:
                M1 = M1 * (list_frac_p[i] - list_frac_p[j])
        return Rational(1,1) / M1

    def extractCoefficientfromMi(self, ip, list_frac_p):
        h = len(list_frac_p)
        nw = h+1
        mi = [None]*(h*nw)
        for i in range(h):
            mi[i*nw + (nw-1)] = self.__Mi(list_frac_p[i], ip, list_frac_p)
            mi[i*nw] = 1
            for j in range(1,h):
                mi[i*nw + j] = list_frac_p[i]**j

        M = Matrix(np.array(mi).reshape(h, nw))
        coeff = linsolve(M, sympy.symbols('a0:{0}'.format(h)))

        return list(coeff.args[0])

    def extractCoefficientfromM(self, list_frac_p):
        h = len(list_frac_p)
        nw = h+1
        mi = [None]*(h*nw)
        for i in range(h):
            mi[i*nw + (nw-1)] = self.__M(list_frac_p[i], list_frac_p) - list_frac_p[i]**h
            mi[i*nw] = 1
            for j in range(1, h):
                mi[i*nw + j] = list_frac_p[i]**j

        M = Matrix(np.array(mi).reshape(h, nw))
        coeff = linsolve(M, sympy.symbols('a0:{0}'.format(h)))

        return list(coeff.args[0])


    def __modToomCook(self, m, r, list_frac_p):
        a = m + r - 1
        print "m={0}, r={1}, p={2}".format(m,r,list_frac_p)
        assert len(list_frac_p) == (a - 1)

        # std::vector<frac> At(a * m);
        self.At = [None]*(a * m)
        self.fAt = list()
        # std::vector<frac> G(a * r);
        self.G = [None]*(a * r)
        self.fG = list()
        # std::vector<frac> Bt(a * a);
        self.Bt = [None]*(a * a)
        self.fBt = list()

        for i in range(m):
            for j in range(a-1):
                self.At[a*i + j] = list_frac_p[j]**i
        for i in range(m-1):
            self.At[a*i + a-1] = 0
        self.At[a*(m-1) + a-1] = 1

        for i in range(a-1):
            for j in range(r):
                self.G[r*i + j] = list_frac_p[i]**j * self.__N(i, list_frac_p)
        for j in range(r-1):
            self.G[(a-1)*r + j] = 0
            self.G[(a-1)*r + (r - 1)] = 1

        for i in range(a-0):
            mi = self.extractCoefficientfromMi(i, list_frac_p)
            for j in range(a-1):
                self.Bt[a*i + j] = mi[j]

        mm = self.extractCoefficientfromM(list_frac_p);
        for i in range(a-1):
            self.Bt[a*i + (a - 1)] = 0
            self.Bt[(a-1)*a + i] = mm[i]

        self.Bt[(a-1)*a + (a-1)] = 1

        for i in range(a*m):
            self.fAt.append(Rational(self.At[i]).evalf())
        for i in range(a*r):
            self.fG.append(Rational(self.G[i]).evalf())
        for i in range(a*a):
            self.fBt.append(Rational(self.Bt[i]).evalf())


    def __multiply(self, m1, m2, m, k, n, i1=False, i2=False):
        m3 = [None]*(m*n)
        for i in range(m):
            for j in range(n):
                acc = 0.
                for x in range(k):
                    acc += m1[i*k + x] * m2[(j*k + x) if i2 else (x*n + j)]
                m3[i*n + j] = acc
        return m3


    def __winograd(self, inp, filts, m, r):
        a = m + r - 1

        # filter transformation
        tmpfG = np.array(self.fG, dtype=np.float32).reshape(a, r)
        tmp = np.dot(tmpfG, filts)
        filtst = np.dot(tmp, tmpfG.T)

        # image transformation
        tmpfBt = np.array(self.fBt, dtype=np.float32).reshape(a,a)
        tmp2 = np.dot(tmpfBt, inp)
        imgt = np.dot(tmp2, tmpfBt.T)

        tmp2 = filtst * imgt

        # output transformation
        tmpfAt = np.array(self.fAt, dtype=np.float32).reshape(m, a)
        tmp3 = np.dot(tmpfAt, tmp2)
        out = np.dot(tmp3, tmpfAt.T)

        return out


    def __evaluateMatrices(self, m, r, list_frac_p, runs):
        a = m + r - 1;
        ref = [None]*(m*m)
        res = [None]*(m*m)

        self.__modToomCook(m, r, list_frac_p)
        #TODO: verify filter

        error = np.array([])
        for rr in range(runs):
            # Generate random inputs.
            inp = np.random.rand(a,a).astype(np.float32)*2-1
            filts = np.random.rand(r,r).astype(np.float32)*2-1

            # Calculate Winograd result.
            res = self.__winograd(inp, filts, m, r)

            # Calculate direct convolution.
            ref = correlate2d(inp.astype(np.float64), filts.astype(np.float64), mode='valid')

            # Compare error.
            error = np.append(error, np.linalg.norm(ref-res, 1))

        return np.mean(error)

    def __find_polynomials(self):
        m = self.out_size
        r = self.filts_size
        RUNS = 1000
        verbose = True
        # input pp
        pc = m + r - 2
        p = [0, 1, -1]

        num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -2, -3, -4, -5, -6, -7, -8, -9]
        den = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        if not self.disable_cache:
	        pp_db_cache = pandas.read_csv('.winograd_pp.cache')
	        if pc+1 in pp_db_cache.alpha.values:
	            p = eval(pp_db_cache[pp_db_cache.alpha==pc+1].min().polynomials)
	        elif pc+1 > pp_db_cache.values[-1][0]:
	            p = eval(pp_db_cache.values[-1][1])

        if len(p)-r+3<2:
            required_pc = 2-3+r
            pp_db_cache = pandas.read_csv('.winograd_pp.cache')
            if required_pc+1 in pp_db_cache.alpha.values:
                print pp_db_cache[pp_db_cache.alpha==required_pc+1].min().polynomials
                p = eval(pp_db_cache[pp_db_cache.alpha==required_pc+1].min().polynomials)
            else:
                print "Cannot start with 3 polynomials when min required pc is: {0}".format(required_pc)
                exit(1)

        print p

        start = time.time()
        for c in range(len(p), pc):
            if (c % 2) == 0:
                if verbose:
                    print "Removing point {0} and add two new.".format(p[-1])
                del p[-1] # pop_back()
                curr = Rational(0)
                error = sys.float_info.max
                for n in range(len(num)/2+1):
                    for d in range(len(den)):
                        t = Rational(num[n], den[d])
                        #frac mt(num[n]*(-1), den[d]);
                        mt = Rational(den[d]*(-1), num[n])
                        if t in p or mt in p:
                            print "Skipping point: {0} and {1}".format(t, mt)
                            continue
                        p.append(t)
                        p.append(mt)
                        e = self.__evaluateMatrices(c-r+3, r, p, RUNS)
                        if verbose:
                            print "Testing\t{0} \t {1}".format(t, e)
                        #std::cout << "PPoints used:\t";
                        #print(p);
                        if e < error:
                            error = e
                            curr = t

                        del p[-1]
                        del p[-1]
                    
                if verbose:
                    print ">> Best pair: (+/-) {0} with error {1}".format(curr, error)
                mt = Rational(curr.q*(-1), curr.p)
                p.append(curr)
                p.append(mt)

            else:
                if verbose:
                    print "Adding one point."

                curr = Rational(0)
                error = sys.float_info.max
                for n in range(len(num)):
                    for d in range(len(den)):
                        t = Rational(num[n], den[d])
                        if t in p:
                            continue
                        p.append(t)
                        e = self.__evaluateMatrices(c-r+3, r, p, RUNS)
                        if verbose:
                            print "Testing {0}\t{1}".format(t, e)

                        if e < error:
                            error = e;
                            curr = t;

                        del p[-1]
                    
                if verbose:
                    print ">> Best point: {0} with error {1}".format(curr, error)
                p.append(curr)

        end = time.time()
        print "Elapsed time (s): {0}".format(end-start)

        return p

    def gen_points(self):
        m = self.out_size
        r = self.filts_size
        p = []

        # pp_db_cache = pandas.read_csv('.winograd_pp.cache', names=['m', 'r', 'p'])
        # if m in pp_db_cache.m.values and r in pp_db_cache.r.values:
        #     p = eval(pp_db_cache.p[(pp_db_cache.m==2) & (pp_db_cache.r==5)].values[0])

        if not p:
            p = self.__find_polynomials()

        print "Selected polynomial points for F({0},{1}): {2}".format(m, r, p)
        self.__modToomCook(m, r, p)
        return p


    def gen_code(self, unroll_factor=1):
        # set array shapes
        G_ndarray = np.asarray(self.fG).reshape(self.alpha, self.filts_size)
        Bt_ndarray = np.asarray(self.fBt).reshape(self.alpha, self.alpha)
        At_ndarray = np.asarray(self.fAt).reshape(self.out_size, self.alpha)
        
        symbolic.main(At_ndarray, G_ndarray, Bt_ndarray, unroll_factor)
        self.Gg =  symbolic.Gg
        self.Gg2 =  symbolic.Gg2
        self.At =  symbolic.At
        self.At2 =  symbolic.At2
        self.Bt =  symbolic.Bt
        self.Bt2 =  symbolic.Bt2


    def test_winograd(self, m, r, p):
        a = m +r -1
        runs = 10000

        error_normal = np.array([])
        error_l1norm = np.array([])
        error_l2norm = np.array([])

        ref = None
        self.__modToomCook(m, r, p)
        for i in range(runs):
            inp = np.random.rand(a,a).astype(np.float32)*2-1
            filts = np.random.rand(r,r).astype(np.float32)*2-1
            
            res = self.__winograd(inp, filts, m, r)
            ref = correlate2d(inp.astype(np.float64), filts.astype(np.float64), mode='valid')

            error_normal = np.append(error_normal, np.sum(np.abs(np.subtract(ref,res))))
            error_l1norm = np.append(error_l1norm, np.linalg.norm(ref-res, 1))
            error_l2norm = np.append(error_l2norm, np.linalg.norm(ref-res))

        print "Error: min={min}, max={max}, median={median}, mean={mean}".format(min=np.min(error_normal), max=np.max(error_normal), median=np.median(error_normal), mean=np.mean(error_normal))
        print "Error l1: min={min}, max={max}, median={median}, mean={mean}, relative={relative}".format(min=np.min(error_l1norm), max=np.max(error_l1norm), median=np.median(error_l1norm), mean=np.mean(error_l1norm), relative=np.median(error_l1norm)/np.linalg.norm(ref,1))
        print "Error l2: min={min}, max={max}, median={median}, mean={mean}, relative={relative}".format(min=np.min(error_l2norm), max=np.max(error_l2norm), median=np.median(error_l2norm), mean=np.mean(error_l2norm), relative=np.median(error_l2norm)/np.linalg.norm(ref))
        
        if not os.path.exists('accuracy_log.csv'):
            with open('accuracy_log.csv', mode='w') as fp:
                csv_fp = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_fp.writerow(['m', 'r', 'alpha', 'polynomials', 'errors'])
        with open('accuracy_log.csv', mode='a') as fp:
            csv_fp = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_fp.writerow([m, r, m+r-1, str([srepr(x) for x in p]).replace("'",""), error_l1norm.tolist()])

        return np.mean(error_l1norm), np.median(error_l1norm)













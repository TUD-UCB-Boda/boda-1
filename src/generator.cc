#include <python2.7/Python.h>
#include <numpy/ndarrayobject.h>
#include "generator.H"
#include <random>
#include <limits>
#include <iostream>
#include <limits>

void print(std::vector<frac> ff) {     for (auto &f : ff)         std::cout << f.str() << " ";     std::cout << std::endl; }

frac tc_gen::M(frac x, std::vector<frac> p)
{
    frac M1(1, 1);
    for (int j = 0; j < p.size(); ++j)
        M1 = M1 * (x - p[j]);
    return M1;
}

frac tc_gen::Mi(frac x, int i, std::vector<frac> p)
{
    frac M1(1, 1);
    for (int j = 0; j < p.size(); ++j)
        if (i != j)
            M1 = M1 * (x - p[j]);
    return M1;
}

frac tc_gen::N(int i, std::vector<frac> p)
{
    frac M1(1, 1);
    for (int j = 0; j < p.size(); ++j)
        if (j != i)
            M1 = M1 * (p[i] - p[j]);
    return frac(1,1) / M1;
}


void tc_gen::solveInplace(std::vector<frac> &mi, int h, int nw)
{
    int i = 0, j = 0;
    frac EPS(1,10000000000);
    frac MEPS(-1,10000000000);
    frac zero = 0;
    while (i < h) {
        bool pivot = false;
        while (j < nw-1 && !pivot) {
            if (mi[i*nw+j] != zero) {
                pivot = true;
            } else {
                long max_i = i;
                frac max_v = 0;
                for (int k = i+1; k < h; ++k) {
                    frac b = mi[k*nw + j] >= zero ? mi[k*nw + j] : mi[k*nw + j] * (-1);
                    if (b > max_i) {
                        max_i = k;
                        max_v = b;
                    }
                }
                if (max_i != i) {
                    // swaprows
                    for (int c = 0; c < nw; ++c) {
                        frac tmp = mi[max_i*nw + c];
                        mi[max_i*nw + c] = mi[i*nw + c];
                        mi[i*nw + c] = tmp;
                    }
                    pivot = true;
                } else
                    ++j;
            }
        }
        if (pivot) {
            for (int t = i + 1; t < h; ++t) {
                for (int s = j +1; s < nw; ++s) {
                    mi[t*nw + s] = mi[t*nw + s] - mi[i*nw + s] * (mi[t*nw + j] / mi[i*nw + j]);
                    if (mi[t*nw + s] < EPS && mi[t*nw + s] > MEPS)
                        mi[t*nw + s] = 0;
                }
                mi[t*nw + j] = 0;
            }
        }
        ++j;
        ++i;
    }
    //std::cout <<"misolve"<<std::endl;
    //print(mi);
    // reducing
    i = h - 1;
    j = nw - 2;

    while (i >= 0) {
        int k = j - 1;
        while (k >= 0) {
            if (mi[i*nw + k] != 0)
                j = k;
            k--;
        }

        if (mi[i*nw + j] != 0) {
            for (int t = i-1; t >= 0; --t) {
                for (int s = 0; s < nw; ++s) {
                    if (s != j) {
                        mi[t*nw + s] = mi[t*nw + s] - mi[i*nw + s] * (mi[t*nw + j] / mi[i*nw + j]);
                        if (mi[t*nw + s] < EPS && mi[t*nw + s] > MEPS)
                            mi[t*nw + s] = 0;
                    }
                }
                mi[t*nw + j] = 0;
            }
            for (int k = j + 1; k < nw; ++k) {
                mi[i*nw +k] = mi[i*nw + k] / mi[i*nw + j];
                if (mi[i*nw + k] < EPS && mi[i*nw + k] > MEPS)
                    mi[i*nw + k] = 0;
            }
            mi[i*nw + j] = 0;
        }
        i--;
        j--;

    }
}

std::vector<frac> tc_gen::extractCoefficientfromMi(int ip, std::vector<frac> p)
{
    const int h = p.size();
    const int nw = h+1;
    std::vector<frac> mi(h*nw);
    for (int i = 0; i < p.size(); ++i) {
        mi[i*nw + (nw-1)] = Mi(p[i], ip, p);
        mi[i*nw] = 1;
        for (int j = 1; j < p.size(); ++j)
            mi[i*nw + j] = p[i].pow(j);
    }
    //print(mi);
    solveInplace(mi, h, nw);
    //print(mi);
    std::vector<frac> coeff(p.size());
    for (int i = 0; i < p.size(); ++i) {
        frac f = mi[i*nw + nw-1];
        //assert(ceilf(f) == f);
        coeff[i] = f;
    }
    //std::cout << "coeff ";
    //print(coeff);
    return coeff;
}
std::vector<frac> tc_gen::extractCoefficientfromM(std::vector<frac> p)
{
    const int h = p.size();
    const int nw = h+1;
    std::vector<frac> mi(h*nw);
    for (int i = 0; i < p.size(); ++i) {
        mi[i*nw + (nw-1)] = M(p[i], p) - p[i].pow(p.size());
        mi[i*nw] = 1;
        for (int j = 1; j < p.size(); ++j)
            mi[i*nw + j] = p[i].pow(j);
    }
    solveInplace(mi, h, nw);
    std::vector<frac> coeff(p.size());
    for (int i = 0; i < p.size(); ++i) {
        frac f = mi[i*nw + nw-1];
        //assert(ceilf(f) == f);
        coeff[i] = f;
    }
    return coeff;
}

//void modToomCook(int m, int r, std::vector<frac> p)
void tc_gen::modToomCook(int m, int r, std::vector<frac> p, float *at, float *bt, float *g, bool printM)
{
    const int a = m + r - 1;
    assert(p.size() == (a - 1));

    std::vector<frac> At(a * m);
    std::vector<frac> G(a * r);
    std::vector<frac> Bt(a * a);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < a-1; ++j)
            At[a*i + j] = p[j].pow(i);
    for (int i = 0; i < m-1; ++i)
        At[a*i + a-1] = 0;
    At[a*(m-1) + a-1] = 1;

    for (int i = 0; i < a - 1; ++i)
        for (int j = 0; j < r; ++j)
            G[r*i + j] = p[i].pow(j) * N(i, p);
    for (int j = 0; j < r - 1; ++j) {
        G[(a-1)*r + j] = 0;
        G[(a-1)*r + (r - 1)] = 1;
    }
    // FIXME for testing:
    /*for (int i = 0; i < r; ++i)
      G[i].neg = !G[i].neg;
*/
    for (int i = 0; i < a - 0; ++i) {
        std::vector<frac> mi = extractCoefficientfromMi(i, p);
        for (int j = 0; j < a - 1; ++j)
            Bt[a*i + j] = mi[j];
    }
    std::vector<frac> mm = extractCoefficientfromM(p);
    //std::cout << print(mm, mm.size()) << std::endl;
    for (int i = 0; i < a - 1; ++i) {
        Bt[a*i + (a - 1)] = 0;
        Bt[(a-1)*a + i] = mm[i];
    }
    Bt[(a-1)*a + (a-1)] = 1;
    // FIXME for testing:
  /*  for (int i = 0; i < a; ++i)
      Bt[i].neg = !Bt[i].neg;
*/

    if (printM) {
        //std::cout << "At:" << std::endl << print(At, a) << std::endl;
        //std::cout << "G:" << std::endl << print(G, r) << std::endl;
        //std::cout << "Bt:" << std::endl << print(Bt, a) << std::endl;
    } else {
        for (int i = 0; i < a*m; ++i)
            fAt.push_back(At[i].toFloat());
        for (int i = 0; i < a*r; ++i)
            fG.push_back(G[i].toFloat());
        for (int i = 0; i < a*a; ++i)
            fBt.push_back(Bt[i].toFloat());
    }
}

void tc_gen::multiply(float *m1, float *m2, float *m3, int m, int k, int n, bool i1, bool i2)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int x = 0; x < k; ++x)
                acc += m1[i*k + x] * m2[i2 ? (j*k + x) : (x*n + j)];
            m3[i*n + j] = acc;
        }
}

void tc_gen::winograd(float *out, float *in, float *filts, int m, int r, float *yt, float *xt, float *w)
{
    int a = m + r - 1;

    // filter transformation
    //std::cout << r << std::endl;
    float *tmp = new float[a*r];
    float *filtst= new float[a*a];
    multiply(w, filts, tmp, a, r, r, false, false);
    //std::cout << "filts_trans_1: " << std::endl;
    multiply(tmp, w, filtst, a, r, a, false, true);
    //std::cout << "trans_filt: " << std::endl;

    // image transformation
    float *tmp2 = new float[a*a];
    float *imgt = new float[a*a];
    multiply(xt, in, tmp2, a, a, a);
    //std::cout << "img_trans_1: " << std::endl;

    multiply(tmp2, xt, imgt, a, a, a, false, true);

    //std::cout << "image_trans: " << std::endl;
    //print_array(imgt, a*a);
    //std::cout << a*a;

    for (int i = 0; i < a*a; ++i)
        tmp2[i] = filtst[i] * imgt[i];


    //std::cout << "gemm: " << std::endl;
    // output transformation
    float *tmp3 = new float[a*m];
    multiply(yt, tmp2, tmp3, m, a, a);
    multiply(tmp3, yt, out, m, a, m, false, true);

    //print_array(out, m*m);
}

double tc_gen::evaluateMatrices(int m, int r, std::vector<frac> &p, int runs)
{
    //r = 3;
    //std::cout << "r:" << r << std::endl;
    const int a = m + r - 1;
    double *ref = new double[m*m];
    float *res = new float[m*m];
    float *in = new float[a*a];
    float *filts = new float[r*r];

    float *at = new float[a*m];
    float *g = new float[a*r];
    float *bt = new float[a*a];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    modToomCook(m, r, p, at, bt, g);
    // TODO verify filter

    double error = 0.f;
    for (int rr = 0; rr < runs; ++rr) {

        // Generate random inputs.
        for (int i = 0; i < a*a; ++i)
            in[i] = dis(gen);
        for (int i = 0; i < r*r; ++i)
            filts[i] = dis(gen);

        // Calculate Winograd result.
        winograd(res, in, filts, m, r, at, bt, g);
        //std::cout << "res:" << std::endl;
        //print_array(res, m*m);

        // Calculate direct convolution.
        for (int oi = 0; oi < m; ++oi)
            for (int oj = 0; oj < m; ++oj) {
                double acc = 0.f;
                for (int i = 0; i < r; ++i)
                    for (int j = 0; j < r; ++j)
                        acc += (double)(in[(i+oi)*a+(j+oj)]) * (double)(filts[i*r+j]);
                ref[oi*m + oj] = acc;
            }

        //print_array(ref, m*m);
        // Compare error.
        for (int i = 0; i < m*m; ++i)
            error += std::abs(ref[i] - res[i]);
    }
    return error / (double)runs;
}


bool tc_gen::existFrac(frac &x, std::vector<frac> &xv)
{
    for (auto &i : xv)
        if (x == i)
            return true;
    return false;
}


void tc_gen::gen_points()
{
    p = {0, 1, -1};
    int m = out_size;
    int r = filts_size;
    const int RUNS = 10000;
    bool verbose = false;

    //input pp
    int pc = m + r - 2;
    if (pc < 4) {
        std::cout << "Polynomial points for F("<<m<<","<<r<<"): ";

        modToomCook(m,r,p, 0,0,0, false);
        return;
    }

    int num[9] = {0, 1, 2, 3, 4, -1, -2, -3, -4};
    int den[4] = {1, 2, 3, 4};

    // FIXME A bug here leads to suboptimal polynomial points. The following hard-coded
    // polynomial points were generated by our "toomcook-generator" script, the initial
    // standalone version of this generator class. 
    if (pc == 4) 
      p = {0, 1, -1, 2};
    if (pc == 5)
      p = {0, 1, -1, frac(1,2), -2};
    if (pc == 6)
      p = {0, 1, -1, frac(1,2), -2, frac(-1,2)};
    if (pc == 7)
      p = {0, 1, -1, frac(1,2), -2, 2, frac(-1,2)};
    if (pc == 8)
      p = {0, 1, -1, frac(1,2), -2, 2, frac(-1,2), 4};
    if (pc == 9)
      p = {0, 1, -1, frac(1,2), -2, 2, frac(-1,2), 3, frac(-1,3)};
    if (pc == 10)
      p = {0, 1, -1, frac(1,2), -2, 2, frac(-1,2), 3, frac(-1,3), 4};

    for (int c = 3; c < pc; ++c)
    {
      break; // FIXME
        if ((c % 2) == 0) {
            if (verbose)
                std::cout << "Removing point "<< p.back().str() << " and add two new." << std::endl;
            p.pop_back();
            frac curr(0);
            double error = std::numeric_limits<double>::max();
            for (int n = 0; n < 5; ++n)
                for (int d = 0; d < 4; ++d) {
                    frac t(num[n], den[d]);
                    //frac mt(num[n]*(-1), den[d]);
                    frac mt(den[d]*(-1), num[n]);
                    if (existFrac(t, p) || existFrac(mt, p)) {
                        //std::cout << "Skipping point: " << t.str() << " and " << mt.str() << ".\n"; continue;}
                        continue;}
                    p.push_back(t);
                    p.push_back(mt);
                    double e = evaluateMatrices(c-r+3, r, p, RUNS);
                    if (verbose)
                        std::cout << "Testing\t" << t.str() << "\t" << e << std::endl;
                    //std::cout << "PPoints used:\t";
                    //print(p);
                    if (e < error) {
                        error = e;
                        curr = t;
                    }
                    p.pop_back();
                    p.pop_back();
                }
            if (verbose)
                std::cout << ">> Best pair: (+/-) " << curr.str() << " with error " << error << std::endl;
            frac mt(curr.den*(-1), curr.num);
            p.push_back(curr);
            p.push_back(mt);
        } else {
            if (verbose)
                std::cout << "Adding one point." << std::endl;
            frac curr(0);
            double error = std::numeric_limits<double>::max();
            for (int n = 0; n < 9; ++n)
                for (int d = 0; d < 4; ++d) {
                    frac t(num[n], den[d]);
                    if (existFrac(t, p))
                        continue;
                    p.push_back(t);
                    double e = evaluateMatrices(c-r+3, r, p, RUNS);
                    if (verbose)
                        std::cout << "Testing\t" << t.str() << "\t" << e << std::endl;
                    //std::cout << "PPoints used:\t";
                    //print(p);
                    if (e < error) {
                        error = e;
                        curr = t;
                    }
                    p.pop_back();
                }
            if (verbose)
                std::cout << ">> Best point: " << curr.str() << " with error " << error << std::endl;
            p.push_back(curr);
        }
    }
    std::cout << "Polynomial points for F("<<m<<","<<r<<"): ";
    print(p);
    std::cout << std::endl;
    modToomCook(m,r,p, 0,0,0, false);
}

// Requires initialized python
std::string tc_gen::get_return(PyObject *pDict, const char *name)
{
    PyObject *pFunc, *pResult;
    pFunc = PyDict_GetItemString(pDict, name);
    if (PyCallable_Check(pFunc)) {
        pResult = PyObject_CallObject(pFunc, 0);
    }
    else
        PyErr_Print();

    std::string ret;
    if (pResult != NULL) {
        ret = PyString_AsString(pResult);
        Py_DECREF(pResult);
    }
    else
        ret = "";
    return ret;
}

void tc_gen::gen_code()
{
    PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *pResult, *pArgs, *pArg;

    // Initialize the Python Interpreter
    Py_Initialize();
    _import_array(); // essential for numpy, random segfaults otherwise

    //PySys_SetArgv(1, {"./m"}); PYTHONPATH=. is important

    // Build the name object
    pName = PyString_FromString("symbolic");

    // Load the module object
    pModule = PyImport_Import(pName);

    // pDict is a borrowed reference
    pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference
    pFunc = PyDict_GetItemString(pDict, "main");

    if (PyCallable_Check(pFunc))
    {
        //int buffer[4] = {1,2,3,4};
        const int a = out_size + filts_size - 1;
        pArgs = PyTuple_New(3);

        //At
        //std::vector<float> buffer(4,2);
        npy_intp dim[2] = {out_size, a};
        PyObject *pat = PyArray_SimpleNewFromData(2, &dim[0], NPY_FLOAT32, &fAt[0]);
        PyTuple_SetItem(pArgs,0,pat);

        npy_intp dim2[2] = {a, filts_size};
        PyObject *pg = PyArray_SimpleNewFromData(2, &dim2[0], NPY_FLOAT32, &fG[0]);
        PyTuple_SetItem(pArgs,1,pg);

        npy_intp dim3[2] = {a, a};
        PyObject *pbt = PyArray_SimpleNewFromData(2, &dim3[0], NPY_FLOAT32, &fBt[0]);
        PyTuple_SetItem(pArgs,2,pbt);

        pResult = PyObject_CallObject(pFunc, pArgs);
    } else
    {
        PyErr_Print();
    }

    if (pResult != NULL) {
        printf("Result of call: %s\n", PyString_AsString(pResult));
        Py_DECREF(pResult);
    } else {
        printf("failed");
    }

    printf("bla: %s\n", get_return(pDict, "getAt").c_str());
    code_Ga = get_return(pDict, "getGg");
    code_Gb = get_return(pDict, "getGg2");
    code_Bta = get_return(pDict, "getBt");
    code_Btb = get_return(pDict, "getBt2");
    code_Ata = get_return(pDict, "getAt");
    code_Atb = get_return(pDict, "getAt2");

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    // Finish the Python Interpreter
    Py_Finalize();
}

void tc_gen::gen_code_py(int ul)
{
    PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *pResult, *pArgs, *pArg;

    // Initialize the Python Interpreter
    Py_Initialize();
    _import_array(); // essential for numpy, random segfaults otherwise

    //PySys_SetArgv(1, {"./m"}); PYTHONPATH=. is important

    // Build the name object
    pName = PyString_FromString("main");

    // Load the module object
    pModule = PyImport_Import(pName);

    // pDict is a borrowed reference
    pDict = PyModule_GetDict(pModule);

    // pFunc is also a borrowed reference
    pFunc = PyDict_GetItemString(pDict, "F");

    if (PyCallable_Check(pFunc))
    {
        //int buffer[4] = {1,2,3,4};
        const int a = out_size + filts_size - 1;
        pArgs = PyTuple_New(4);

        PyObject *pValue = PyInt_FromLong(out_size);
        PyTuple_SetItem(pArgs, 0, pValue);

        pValue = PyInt_FromLong(filts_size);
        PyTuple_SetItem(pArgs, 1, pValue);

        pValue = PyInt_FromLong(ul);
        PyTuple_SetItem(pArgs, 2, pValue);

        pValue = PyInt_FromLong(0);
        PyTuple_SetItem(pArgs, 3, pValue);


        //At
        //std::vector<float> buffer(4,2);
/*        npy_intp dim[2] = {out_size, a};
        PyObject *pat = PyArray_SimpleNewFromData(2, &dim[0], NPY_FLOAT32, &fAt[0]);
        PyTuple_SetItem(pArgs,0,pat);

        npy_intp dim2[2] = {a, filts_size};
        PyObject *pg = PyArray_SimpleNewFromData(2, &dim2[0], NPY_FLOAT32, &fG[0]);
        PyTuple_SetItem(pArgs,1,pg);

        npy_intp dim3[2] = {a, a};
        PyObject *pbt = PyArray_SimpleNewFromData(2, &dim3[0], NPY_FLOAT32, &fBt[0]);
        PyTuple_SetItem(pArgs,2,pbt);*/

        pResult = PyObject_CallObject(pFunc, pArgs);
    } else
    {
        PyErr_Print();
    }

   /* if (pResult != NULL) {
        printf("Result of call: %s\n", PyString_AsString(pResult));
        Py_DECREF(pResult);
    } else {
        printf("failed");
    }*/

    printf("bla: %s\n", get_return(pDict, "getGg").c_str());
    code_Ga = get_return(pDict, "getGg");
    code_Gb = get_return(pDict, "getGg2");
    code_Bta = get_return(pDict, "getBt");
    code_Btb = get_return(pDict, "getBt2");
    code_Ata = get_return(pDict, "getAt");
    code_Atb = get_return(pDict, "getAt2");

    // Clean up
    Py_DECREF(pModule);
    Py_DECREF(pName);

    // Finish the Python Interpreter
    Py_Finalize();
}

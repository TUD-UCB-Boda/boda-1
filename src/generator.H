#ifndef TCGEN_H
#define TCGEN_H

#include <numpy/ndarrayobject.h>
#include "fraction.H"
#include <vector>

// class frac;

class tc_gen
{
    uint32_t out_size, filts_size;
    std::vector<frac> p;

    std::vector<frac> G, Bt, At;
    std::vector<float>fG, fBt, fAt;

    private:
        frac M(frac x, std::vector<frac> p);
        frac Mi(frac x, int i, std::vector<frac> p);
        frac N(int i, std::vector<frac> p);
        void solveInplace(std::vector<frac> &mi, int h, int nw);
        std::vector<frac> extractCoefficientfromMi(int ip, std::vector<frac> p);
        std::vector<frac> extractCoefficientfromM(std::vector<frac> p);
        void modToomCook(int m, int r, std::vector<frac> p, float *at, float *bt, float *g, bool printM = false);
        void multiply(float *m1, float *m2, float *m3, int m, int k, int n, bool i1 = false, bool i2 = false);
        void winograd(float *out, float *in, float *filts, int m, int r, float *yt, float *xt, float *w);
        double evaluateMatrices(int m, int r, std::vector<frac> &p, int runs);
        bool existFrac(frac &x, std::vector<frac> &xv);
        std::string get_return(PyObject *pDict, const char *name);

    public:
        tc_gen(uint32_t out_size, uint32_t filts_size)
            : out_size(out_size), filts_size(filts_size) {}

        void set_geometry(uint32_t out_size, uint32_t filts_size);
        void gen_points();
        void gen_code();
        void gen_code_py(int ul);

    std::string code_Ga, code_Gb, code_Bta, code_Btb, code_Ata, code_Atb;
};


#endif

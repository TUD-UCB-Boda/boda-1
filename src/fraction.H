#ifndef FRACTION_H
#define FRACTION_H

#include <numeric>
#include <string>
#include <sstream>
#include <cmath>
#include <assert.h>
#define ABS(v) ((v < 0) ? 0-v : v)
#define NEG(n,v) (n ? 0-v : v)




struct frac
{
    long num;
    long den;
    bool neg;

    long gcd ( long a, long b )
    {
      long c;
      while ( a != 0 ) {
         c = a; a = b%a;  b = c;
      }
      return b;
    }

    frac() { }
    frac(char * c)
    {
        neg = false;
        std::string cs(c);
        std::string del = "/";
        size_t find = cs.find(del);
        if (find != std::string::npos) {
            num = std::stoi(cs.substr(0, find));
            den = std::stoi(cs.substr(find + 1));
        } else {
            num = std::stoi(cs);
            den = 1;
        }
        if (num < 0) {
            num *= -1;
            neg = true;
        }
    }
    frac(long n, long d, bool neg)
    {
        num = ABS(n);
        den = ABS(d);
        this->neg = neg;
        simplify();
    }

    frac(long n, long d)
    {
        num = ABS(n);
        den = ABS(d);
        neg = (n * d) < 0;
        simplify();
    }

    frac(int n) : frac(n, 1) {}

    frac operator*(const frac& rhs)
    {
        return frac(this->num * rhs.num, this->den * rhs.den, this->neg ^ rhs.neg);
    }

    frac operator+(const frac& rhs)
    {
        return frac(NEG(this->neg, this->num) * rhs.den + this->den * NEG(rhs.neg, rhs.num), this->den * rhs.den);
    }

    frac operator-(const frac& rhs)
    {
        return frac(NEG(this->neg, this->num) * rhs.den - this->den * NEG(rhs.neg, rhs.num), this->den * rhs.den);
    }

    frac operator/(const frac& rhs)
    {
        return frac(NEG(this->neg, this->num) * rhs.den, NEG(rhs.neg, rhs.num) * this->den);
    }

    friend bool operator<(frac& l, long& r)
    {
        frac t(r);
        return (l - t).neg;
    }

    friend bool operator>(frac& l, long& r)
    {
        frac t(r);
        return t < l;
    }
    friend bool operator<(frac& l, frac& r)
    {
        return (l - r).neg;
    }

    friend bool operator>(frac& l, frac& r)
    {
        return r < l;
    }

    friend bool operator>=(frac& l, frac& r)
    {
        return !(l < r);
    }

    friend bool operator<=(frac& l, frac& r)
    {
        return !(l > r);
    }

    friend bool operator==(const frac& l, const frac& r)
    {
        return l.num == r.num &&
            (l.num == 0 || ((l.den == r.den) && (l.neg == r.neg)));
    }

    friend bool operator!=(const frac& l, const frac& r)
    {
        return !(l == r);
    }

    frac& operator=(frac f) noexcept
    {
        num = f.num;
        den = f.den;
        neg = f.neg;
        return *this;
    }

    frac& operator=(long f) noexcept
    {
        num = ABS(f);
        den = 1;
        neg = f < 0;
        return *this;
    }
    std::string str()
    {
        if (num == 0)
            return std::string("0");
        std::ostringstream oss;
        if (neg)
            oss << "-";
        oss << num;
        if (den != 1)
            oss << "/" << den;
        return oss.str();
    }

    void simplify()
    {
        long div = gcd(num, den);
        if (div != 0) {
            num /= div;
            den /= div;
        }
    }

    frac pow(int p)
    {
        frac f(1,1);
        long n = std::pow(NEG(neg, num), p);
        f.num = ABS(n);
        f.den = std::pow(den, p);
        f.neg = (n < 0);
        f.simplify();
        return f;
    }

    float toFloat()
    {
        if (num == 0)
            return 0.f;
        return (float)num / (float)den * (neg ? -1.f : 1.f);
    }

};

#endif

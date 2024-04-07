#pragma once

#if defined(__cplusplus)
    #include <cstring>
#endif

struct __half_raw {
    unsigned short x;
};

struct __half2_raw {
    unsigned short x;
    unsigned short y;
};

#if defined(__cplusplus)
    struct __half;

    __half __float2half(float);
    float __half2float(__half);

    // BEGIN STRUCT __HALF
    struct __half {
    protected:
        unsigned short __x;
    public:
        // CREATORS
        __half() = default;
        __half(const __half_raw& x) : __x{x.x} {}
        #if !defined(__HIP_NO_HALF_CONVERSIONS__)
            __half(float x) : __x{__float2half(x).__x} {}
            __half(double x) : __x{__float2half(x).__x} {}
        #endif
        __half(const __half&) = default;
        __half(__half&&) = default;
        ~__half() = default;

        // MANIPULATORS
        __half& operator=(const __half&) = default;
        __half& operator=(__half&&) = default;
        __half& operator=(const __half_raw& x) { __x = x.x; return *this; }
        #if !defined(__HIP_NO_HALF_CONVERSIONS__)
            __half& operator=(float x)
            {
                __x = __float2half(x).__x;
                return *this;
            }
            __half& operator=(double x)
            {
                return *this = static_cast<float>(x);
            }
        #endif

        // ACCESSORS
        operator float() const { return __half2float(*this); }
        operator __half_raw() const { return __half_raw{__x}; }
    };
    // END STRUCT __HALF

    // BEGIN STRUCT __HALF2
    struct __half2 {
    public:
        __half x;
        __half y;

        // CREATORS
        __half2() = default;
        __half2(const __half2_raw& ix)
            :
            x{reinterpret_cast<const __half&>(ix.x)},
            y{reinterpret_cast<const __half&>(ix.y)}
        {}
        __half2(const __half& ix, const __half& iy) : x{ix}, y{iy} {}
        __half2(const __half2&) = default;
        __half2(__half2&&) = default;
        ~__half2() = default;

        // MANIPULATORS
        __half2& operator=(const __half2&) = default;
        __half2& operator=(__half2&&) = default;
        __half2& operator=(const __half2_raw& ix)
        {
            x = reinterpret_cast<const __half_raw&>(ix.x);
            y = reinterpret_cast<const __half_raw&>(ix.y);
            return *this;
        }

        // ACCESSORS
        operator __half2_raw() const
        {
            return __half2_raw{
                reinterpret_cast<const unsigned short&>(x),
                reinterpret_cast<const unsigned short&>(y)};
        }
    };
    // END STRUCT __HALF2

    inline
    unsigned short __internal_float2half(
        float flt, unsigned int& sgn, unsigned int& rem)
    {
        unsigned int x{};
        std::memcpy(&x, &flt, sizeof(flt));

        unsigned int u = (x & 0x7fffffffU);
        sgn = ((x >> 16) & 0x8000U);

        // NaN/+Inf/-Inf
        if (u >= 0x7f800000U) {
            rem = 0;
            return static_cast<unsigned short>(
                (u == 0x7f800000U) ? (sgn | 0x7c00U) : 0x7fffU);
        }
        // Overflows
        if (u > 0x477fefffU) {
            rem = 0x80000000U;
            return static_cast<unsigned short>(sgn | 0x7bffU);
        }
        // Normal numbers
        if (u >= 0x38800000U) {
            rem = u << 19;
            u -= 0x38000000U;
            return static_cast<unsigned short>(sgn | (u >> 13));
        }
        // +0/-0
        if (u < 0x33000001U) {
            rem = u;
            return static_cast<unsigned short>(sgn);
        }
        // Denormal numbers
        unsigned int exponent = u >> 23;
        unsigned int mantissa = (u & 0x7fffffU);
        unsigned int shift = 0x7eU - exponent;
        mantissa |= 0x800000U;
        rem = mantissa << (32 - shift);
        return static_cast<unsigned short>(sgn | (mantissa >> shift));
    }

    inline
    __half __float2half(float x)
    {
        __half_raw r;
        unsigned int sgn{};
        unsigned int rem{};
        r.x = __internal_float2half(x, sgn, rem);
        if (rem > 0x80000000U || (rem == 0x80000000U && (r.x & 0x1))) ++r.x;

        return r;
    }

    inline
    __half __float2half_rn(float x) { return __float2half(x); }

    inline
    __half __float2half_rz(float x)
    {
        __half_raw r;
        unsigned int sgn{};
        unsigned int rem{};
        r.x = __internal_float2half(x, sgn, rem);

        return r;
    }

    inline
    __half __float2half_rd(float x)
    {
        __half_raw r;
        unsigned int sgn{};
        unsigned int rem{};
        r.x = __internal_float2half(x, sgn, rem);
        if (rem && sgn) ++r.x;

        return r;
    }

    inline
    __half __float2half_ru(float x)
    {
        __half_raw r;
        unsigned int sgn{};
        unsigned int rem{};
        r.x = __internal_float2half(x, sgn, rem);
        if (rem && !sgn) ++r.x;

        return r;
    }

    inline
    __half2 __float2half2_rn(float x)
    {
        return __half2{__float2half_rn(x), __float2half_rn(x)};
    }

    inline
    __half2 __floats2half2_rn(float x, float y)
    {
        return __half2{__float2half_rn(x), __float2half_rn(y)};
    }

    inline
    float __internal_half2float(unsigned short x)
    {
        unsigned int sign = ((x >> 15) & 1);
        unsigned int exponent = ((x >> 10) & 0x1f);
        unsigned int mantissa = ((x & 0x3ff) << 13);

        if (exponent == 0x1fU) { /* NaN or Inf */
            mantissa = (mantissa ? (sign = 0, 0x7fffffU) : 0);
            exponent = 0xffU;
        } else if (!exponent) { /* Denorm or Zero */
            if (mantissa) {
                unsigned int msb;
                exponent = 0x71U;
                do {
                    msb = (mantissa & 0x400000U);
                    mantissa <<= 1; /* normalize */
                    --exponent;
                } while (!msb);
                mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
            }
        } else {
            exponent += 0x70U;
        }
        unsigned int u = ((sign << 31) | (exponent << 23) | mantissa);
        float f;
        memcpy(&f, &u, sizeof(u));

        return f;
    }

    inline
    float __half2float(__half x)
    {
        return __internal_half2float(static_cast<__half_raw>(x).x);
    }

    inline
    float __low2float(__half2 x)
    {
        return __internal_half2float(static_cast<__half2_raw>(x).x);
    }

    inline
    float __high2float(__half2 x)
    {
        return __internal_half2float(static_cast<__half2_raw>(x).y);
    }

    #if !defined(HIP_NO_HALF)
        using half = __half;
        using half2 = __half2;
    #endif
#endif // defined(__cplusplus)

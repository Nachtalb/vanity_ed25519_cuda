#include "ge.cuh"
#include "precomp_data.cuh"

__device__ void ge_p2_0(ge_p2 *h) {
    fe_0(h->X); fe_1(h->Y); fe_1(h->Z);
}

__device__ void ge_p3_0(ge_p3 *h) {
    fe_0(h->X); fe_1(h->Y); fe_1(h->Z); fe_0(h->T);
}

__device__ void ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
}

// This was the cause of the error. Now it is visible.
__device__ void ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
    fe_mul(r->T, p->X, p->Y);
}

// --------------------------------------------------------------------------
// VARIABLE TIME LOOKUP
// --------------------------------------------------------------------------
__device__ void select_vartime(ge_precomp *t, int pos, signed char b) {
    if (b == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
        return;
    }

    unsigned char babs = (b < 0) ? -b : b;
    int idx = babs - 1;

    // Direct read from global constant memory
    const ge_precomp *entry = &base[pos][idx];

    fe_copy(t->yplusx, entry->yplusx);
    fe_copy(t->yminusx, entry->yminusx);
    fe_copy(t->xy2d, entry->xy2d);

    if (b < 0) {
        fe tmp;
        fe_copy(tmp, t->yplusx);
        fe_copy(t->yplusx, t->yminusx);
        fe_copy(t->yminusx, tmp);
        fe_neg(t->xy2d, t->xy2d);
    }
}

// --------------------------------------------------------------------------
// GROUP OPERATIONS
// --------------------------------------------------------------------------

__device__ void ge_add(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->YplusX);
    fe_mul(r->Y, r->Y, q->YminusX);
    fe_mul(r->T, q->T2d, p->T);
    fe_mul(r->X, p->Z, q->Z);
    fe_add(t0, r->X, r->X);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}

__device__ void ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yplusx);
    fe_mul(r->Y, r->Y, q->yminusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}

__device__ void ge_msub(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yminusx);
    fe_mul(r->Y, r->Y, q->yplusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_sub(r->Z, t0, r->T);
    fe_add(r->T, t0, r->T);
}

__device__ void ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p) {
    fe t0;
    fe_sq(r->X, p->X);
    fe_sq(r->Z, p->Y);
    fe_sq2(r->T, p->Z);
    fe_add(r->Y, p->X, p->Y);
    fe_sq(t0, r->Y);
    fe_add(r->Y, r->Z, r->X);
    fe_sub(r->Z, r->Z, r->X);
    fe_sub(r->X, t0, r->Y);
    fe_sub(r->T, r->T, r->Z);
}

__device__ void ge_p3_to_p2(ge_p2 *r, const ge_p3 *p) {
    fe_copy(r->X, p->X);
    fe_copy(r->Y, p->Y);
    fe_copy(r->Z, p->Z);
}

__device__ void ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p) {
    ge_p2 q;
    ge_p3_to_p2(&q, p);
    ge_p2_dbl(r, &q);
}

// --------------------------------------------------------------------------
// MAIN SCALAR MULTIPLICATION (Optimized)
// --------------------------------------------------------------------------

__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a) {
    signed char e[64];
    signed char carry;
    ge_p1p1 r;
    ge_p2 s;
    ge_precomp t;
    int i;

    for (i = 0; i < 32; ++i) {
        e[2 * i + 0] = (a[i] >> 0) & 15;
        e[2 * i + 1] = (a[i] >> 4) & 15;
    }
    carry = 0;
    for (i = 0; i < 63; ++i) {
        e[i] += carry;
        carry = e[i] + 8;
        carry >>= 4;
        e[i] -= carry << 4;
    }
    e[63] += carry;

    ge_p3_0(h);

    #pragma unroll
    for (i = 1; i < 64; i += 2) {
        select_vartime(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }

    ge_p3_dbl(&r, h);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p3(h, &r);

    #pragma unroll
    for (i = 0; i < 64; i += 2) {
        select_vartime(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

// --------------------------------------------------------------------------
// TO BYTES (Output)
// --------------------------------------------------------------------------

__device__ void ge_tobytes(unsigned char *s, const ge_p2 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

// --------------------------------------------------------------------------
// COMPATIBILITY HELPERS
// --------------------------------------------------------------------------
__constant__ fe d2 = { -21827239, -5839606, -30745221, 13898782, 229458, 15978800, -12551817, -6495438, 29715968, 9444199 };

__device__ void ge_p3_to_cached(ge_cached *r, const ge_p3 *p) {
    fe_add(r->YplusX, p->Y, p->X);
    fe_sub(r->YminusX, p->Y, p->X);
    fe_copy(r->Z, p->Z);
    fe_mul(r->T2d, p->T, d2);
}

__device__ void ge_sub(ge_p1p1 *r, const ge_p3 *p, const ge_cached *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->YminusX);
    fe_mul(r->Y, r->Y, q->YplusX);
    fe_mul(r->T, q->T2d, p->T);
    fe_mul(r->X, p->Z, q->Z);
    fe_add(t0, r->X, r->X);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_sub(r->Z, t0, r->T);
    fe_add(r->T, t0, r->T);
}

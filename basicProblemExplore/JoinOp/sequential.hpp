#pragma once
#include <stdint.h>
using JoinFunc = int (*)(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb);

int outerJoin(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb)
{
    int pa, pb, pc;
    pa = pb = pc = 0;
    while (pa < lenA && pb < lenB)
    {
        auto ka = a[pa];
        auto kb = b[pb];
        if (ka == kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = vb[pb];
            c[pc] = ka;
            pa++;
            pb++;
            pc++;
        }
        else if (ka < kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = db;
            c[pc] = a[pa];
            pa++;
            pc++;
        }
        else
        {
            vca[pc] = da;
            vcb[pc] = vb[pb];
            c[pc] = b[pb];
            pb++;
            pc++;
        }
    }
    while (pa < lenA)
    {
        vca[pc] = va[pa];
        vcb[pc] = db;
        c[pc] = a[pa];
        pa++;
        pc++;
    }
    while (pb < lenB)
    {
        vca[pc] = da;
        vcb[pc] = vb[pb];
        c[pc] = b[pb];
        pb++;
        pc++;
    }
    return pc;
}

// da,db not used. Keep them to make all interface idential
int innerJoin(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb)
{
    int pa, pb, pc;
    pa = pb = pc = 0;
    while (pa < lenA && pb < lenB)
    {
        auto ka = a[pa];
        auto kb = b[pb];
        if (ka == kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = vb[pb];
            c[pc] = ka;
            pa++;
            pb++;
            pc++;
        }
        else if (ka < kb)
        {
            pa++;
        }
        else
        {
            pb++;
        }
    }
    return pc;
}
int xorJoin(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb)
{
    int pa, pb, pc;
    pa = pb = pc = 0;
    while (pa < lenA && pb < lenB)
    {
        auto ka = a[pa];
        auto kb = b[pb];
        if (ka == kb)
        {
            pa++;
            pb++;
        }
        else if (ka < kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = db;
            c[pc] = a[pa];
            pa++;
            pc++;
        }
        else
        {
            vca[pc] = da;
            vcb[pc] = vb[pb];
            c[pc] = b[pb];
            pb++;
            pc++;
        }
    }
    while (pa < lenA)
    {
        vca[pc] = va[pa];
        vcb[pc] = db;
        c[pc] = a[pa];
        pa++;
        pc++;
    }
    while (pb < lenB)
    {
        vca[pc] = da;
        vcb[pc] = vb[pb];
        c[pc] = b[pb];
        pb++;
        pc++;
    }
    return pc;
}
// da not used. Keep it to make all interface idential
int diffJoin(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb)
{
    int pa, pb, pc;
    pa = pb = pc = 0;
    while (pa < lenA && pb < lenB)
    {
        auto ka = a[pa];
        auto kb = b[pb];
        if (ka == kb)
        {
            pa++;
            pb++;
        }
        else if (ka < kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = db;
            c[pc] = a[pa];
            pa++;
            pc++;
        }
        else
        {
            pb++;
        }
    }
    while (pa < lenA)
    {
        vca[pc] = va[pa];
        vcb[pc] = db;
        c[pc] = a[pa];
        pa++;
        pc++;
    }
    return pc;
}

int leftJoin(
    uint32_t *a, uint32_t *va, uint32_t da, int lenA,
    uint32_t *b, uint32_t *vb, uint32_t db, int lenB,
    uint32_t *c, uint32_t *vca, uint32_t *vcb)
{
    int pa, pb, pc;
    pa = pb = pc = 0;
    while (pa < lenA && pb < lenB)
    {
        auto ka = a[pa];
        auto kb = b[pb];
        if (ka == kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = vb[pb];
            c[pc] = ka;
            pa++;
            pb++;
            pc++;
        }
        else if (ka < kb)
        {
            vca[pc] = va[pa];
            vcb[pc] = db;
            c[pc] = a[pa];
            pa++;
            pc++;
        }
        else
        {
            pb++;
        }
    }
    while (pa < lenA)
    {
        vca[pc] = va[pa];
        vcb[pc] = db;
        c[pc] = a[pa];
        pa++;
        pc++;
    }
    return pc;
}

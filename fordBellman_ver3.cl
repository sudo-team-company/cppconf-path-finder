#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable
#define WORKGROUP 64
#define INF 1e12

__kernel __attribute__((reqd_work_group_size(WORKGROUP, 1, 1)))
void bellmanFordInit(__global double* restrict d, uint len, uint sourceId)
{
    uint edgeId = get_global_id(0);

    if (edgeId < len) {
        d[edgeId] = (edgeId == sourceId ? 0 : INF);
    }
}

inline double atomic_load(__global double* ptr)
{
    return as_double(atom_and((__global ulong*)ptr, 0xfffffffffffffffful));
}

inline void atomic_min_r64(__global double* target, double value)
{
    double desired;

    double old = atomic_load(target);

    if (old < value) return;

    do {
        old = atomic_load(target);
        desired = old < value ? old : value;
    } while (atom_cmpxchg((__global ulong*)target, as_ulong(old), as_ulong(desired)) != as_ulong(old));
}


__kernel __attribute__((reqd_work_group_size(WORKGROUP, 1, 1)))
void bellmanFordIter(
    uint nEdges,  
    __global const uint2* restrict edges, 
    __global const double* restrict weights,
    __global double* restrict d, 
    __global uint* restrict changed
)
{
    uint edgeId = get_global_id(0);

    if (edgeId >= nEdges) {
        return;
    }

    if (d[edges[edgeId].s0] < INF) {
        double relaxWeight = d[edges[edgeId].s0] + weights[edgeId];

        if (d[edges[edgeId].s1] > relaxWeight) {
            atomic_min_r64(&d[edges[edgeId].s1], relaxWeight);
            *changed = 1;
        }
    }
}

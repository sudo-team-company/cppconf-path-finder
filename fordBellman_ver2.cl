#define WORKGROUP 64
#define INF 1e12

__kernel __attribute__((reqd_work_group_size(WORKGROUP, 1, 1))) void
bellmanFordInit(__global double *restrict d, uint len, uint sourceId) {
  uint edgeId = get_global_id(0);

  if (edgeId < len) {
    d[edgeId] = (edgeId == sourceId ? 0 : INF);
  }
}

__kernel __attribute__((reqd_work_group_size(WORKGROUP, 1, 1))) void
bellmanFordIter(uint nEdges, __global const uint2 *restrict edges,
                __global const double *restrict weights,
                __global double *restrict d, __global uint *restrict changed) {
  uint edgeId = get_global_id(0);
  uint global_size = get_global_size(0);
  uint width = (nEdges + global_size - 1) / global_size;

  uint l = edgeId * width;
  if (l >= nEdges) {
    return;
  }

  uint r = l + width;
  if (r >= nEdges) {
    r = nEdges;
  }

  for (int i = l; i < r; i++) {
    if (d[edges[i].x] < INF) {
      double relaxWeight = d[edges[i].x] + weights[i];

      if (d[edges[i].y] > relaxWeight) {
        d[edges[i].y] = relaxWeight;
        *changed = 1;
      }
    }
  }
}

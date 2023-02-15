#include <omp.h>
#include <stdio.h>

#define BLOCKSIZE 1024.0

__device__ double sqr(float value) {
  return value * value;
}
// TODO: Rewrite and add batching support
__global__ void findNeighbors(float3 *points, float3 *queries, double *sums, const double epsilon, const int numPoints, const int numQueries, const int limit) {
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= numQueries) {
		return;
	}

	float3 query = queries[tid];

  float3 point;
  double distance;
  for (int i = 0; i < numPoints; i++) {
    distance = 0;
    point = points[i];
    if (point.x == query.x && point.y == query.y && point.z == query.z) {
      continue;
    }
    // printf("Query %u: [%f, %f, %f], Point %u: [%f, %f, %f], distance: %f\n", tid, query[dim].x, query[dim].y, query[dim].z, i, point.x, point.y, point.z, dot(query[dim] - point, query[dim] - point));
    distance += sqr(point.x - query.x);
    distance += sqr(point.y - query.y);
    distance += sqr(point.z - query.z);

    if (distance < epsilon * epsilon) {
      sums[tid] += sqrt(distance);
    }
  }
}

double bruteForceSearch(float3 *points, float3 *queries, const double epsilon, const int numPoints, const int numQueries, const int limit) {
  double total_time = omp_get_wtime();
  double *sums;
  float3 *d_points;
  float3 *d_queries;

  cudaMalloc((void **) &d_points, numPoints  * sizeof(float3));
  cudaMalloc((void **) &d_queries, numQueries * sizeof(float3));
  cudaMemcpy(d_points, points, numPoints * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, queries, numQueries * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMallocManaged((void **) &sums, numQueries * numPoints * sizeof(double));

  double search_time = omp_get_wtime();
  const unsigned int totalBlocks = ceil(numQueries * 1.0 / BLOCKSIZE);
  findNeighbors<<<totalBlocks, BLOCKSIZE>>>(d_points, d_queries, sums, epsilon, numPoints, numQueries, limit);
  cudaDeviceSynchronize();
  search_time = omp_get_wtime() - search_time;

  double totalSum = 0;
  for (int i = 0; i < numQueries * numPoints; i++) {
    totalSum += sums[i];
  }
  
  cudaFree(d_points);
  cudaFree(d_queries);
  cudaFree(sums);
  total_time = omp_get_wtime() - total_time - search_time;
  printf("time brute force overhead: %f\n", total_time);
  printf("Brute Force sum: %f\n", totalSum);
  return totalSum;
}

__global__ void warmup(unsigned int *tmp) {
  if (threadIdx.x == 0)
    *tmp = 555;

  return;
}

/**
 * Warms up each GPU by allocating memory on the device, running the kernel warmup, and copying data
 * back to the host.
 *
 * @param device - the device the warm up.
 */
void warm_up_gpu(const int device) {
  cudaSetDevice(device);
  unsigned int *dev_tmp;
  unsigned int *tmp;
  tmp = (unsigned int *) malloc(sizeof(unsigned int));
  *tmp = 0;
  cudaMalloc((unsigned int **) &dev_tmp, sizeof(unsigned int));

  warmup<<<1, 256>>>(dev_tmp);

  // copy data from device to host
  cudaMemcpy(tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(dev_tmp);
  free(tmp);
}
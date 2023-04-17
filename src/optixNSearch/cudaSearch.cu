#include <omp.h>
#include <stdio.h>

#define BLOCKSIZE 1024.0
#define MAXDIMS 30

__host__ __device__ static inline int get3DIndex(int i, int j, int k, int j_size, int k_size) {
  return i*j_size*k_size + j*k_size + k;
}

__host__ __device__ static inline int get2DIndex(int i, int j, int j_size) {
  return i*j_size + j;
}

__host__ void flattenArray(float3 **in, float3 *out, int i_size, int j_size);

__device__ double sqr(float value) {
  return value * value;
}

void flattenArray(float3 **in, float3 *out, int i_size, int j_size) {
  for (int i = 0; i < i_size; i++) {
    for (int j = 0; j < j_size; j++) {
      out[get2DIndex(i, j, j_size)] = in[i][j];
    }
  }
}

// TODO: Rewrite and add batching support
__global__ void findNeighbors(float3 *points, float3 *queries, double *sums, unsigned long *numNeighbors, const double epsilon, const int numPoints, const int numQueries, const int limit, const int numDims) {
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	if (tid >= numQueries) {
		return;
  }
	
  float3 query[MAXDIMS];
  for (int dim = 0; dim < numDims / 3; dim++) {
    query[dim] = queries[get2DIndex(dim, tid, numQueries)];
  }

  float3 point;
  double distance;
  for (int i = 0; i < numPoints; i++) {
    distance = 0;
    for (int dim = 0; dim < numDims / 3; dim++) {
      point = points[get2DIndex(dim, i, numPoints)];
      if (point.x == query[dim].x && point.y == query[dim].y && point.z == query[dim].z) {
        continue;
      }
      // printf("Query %u: [%f, %f, %f], Point %u: [%f, %f, %f], distance: %f\n", tid, query[dim].x, query[dim].y, query[dim].z, i, point.x, point.y, point.z, dot(query[dim] - point, query[dim] - point));
      distance += sqr(point.x - query[dim].x);
      distance += sqr(point.y - query[dim].y);
      distance += sqr(point.z - query[dim].z);
      if (distance >= epsilon * epsilon) {
        continue;
      }
    }
    if (distance < epsilon * epsilon) {
      // printf("Query %u, Point %u, distance: %f\n", tid, i, sqrt(distance));
      sums[tid] += sqrt(distance);
      numNeighbors[tid]++;
    }
  }
}

// TODO: K: add in storing neighbor pairs for a fairer comparison
double bruteForceSearch(float3 **points, float3 **queries, const double epsilon, const int numPoints, const int numQueries, const int limit, const int numDims) {
  double total_time = omp_get_wtime();
  double *sums;
  unsigned long *numNeighbors;
  float3 *d_points;
  float3 *d_queries;
  float3 *flattenedPoints = (float3 *) malloc(numPoints * numDims * sizeof(float3));
  float3 *flattenedQueries = (float3 *) malloc(numQueries * numDims * sizeof(float3));
  flattenArray(points, flattenedPoints, numDims / 3, numPoints);
  flattenArray(queries, flattenedQueries, numDims / 3, numQueries);

  cudaMalloc((void **) &d_points, numPoints * numDims * sizeof(float3));
  cudaMalloc((void **) &d_queries, numQueries * numDims * sizeof(float3));
  cudaMemcpy(d_points, flattenedPoints, numPoints * numDims * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, flattenedQueries, numQueries * numDims * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMallocManaged((void **) &sums, numQueries * sizeof(double));
  cudaMallocManaged((void **) &numNeighbors, numQueries * sizeof(double));

  double search_time = omp_get_wtime();
  const unsigned int totalBlocks = ceil(numQueries * 1.0 / BLOCKSIZE);
  findNeighbors<<<totalBlocks, BLOCKSIZE>>>(d_points, d_queries, sums, numNeighbors, epsilon, numPoints, numQueries, limit, numDims);
  cudaDeviceSynchronize();
  search_time = omp_get_wtime() - search_time;

  double totalSum = 0;
  unsigned long totalNeighbors = 0;
  for (int i = 0; i < numQueries; i++) {
    totalSum += sums[i];
    totalNeighbors += numNeighbors[i];
  }
  
  cudaFree(d_points);
  cudaFree(d_queries);
  cudaFree(sums);
  cudaFree(numNeighbors);
  total_time = omp_get_wtime() - total_time - search_time;
  printf("time brute force overhead: %f\n", total_time);
  printf("Brute Force sum: %f\n", totalSum);
  printf("Number of neighbors: %lu\n", totalNeighbors);
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
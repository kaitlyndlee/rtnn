#include <stdio.h>
#include "state.h"

__device__ double sqr(float value);
__device__ int isIn(unsigned int *in, unsigned int value, size_t in_size);
__global__ void findClosestPoints(float3 *points, float3 *queries, unsigned int *check, double *sums, const double epsilon, const int numDims, const int numPoints, const int numQueries, const int limit);
__global__ void findClosestPointsBruteForce(float3 *points, float3 *queries, double *sums, const double epsilon, const int numDims, const int numPoints, const int numQueries, const int limit);

void flattenArray(float3 **in, float3 *out, int i_size, int j_size);

__host__ __device__ static inline int get3DIndex(int i, int j, int k, int j_size, int k_size) {
  return i*j_size*k_size + j*k_size + k;
}

__host__ __device__ static inline int get2DIndex(int i, int j, int j_size) {
  return i*j_size + j;
}

void calcIntersection(unsigned int **data, 
                      unsigned int *result, 
                      const int numDims, 
                      const int numQueries, 
                      const int limit) {
  unsigned int value;
  int check = 0;
  int write_index = 0;
  for (int q = 0; q < numQueries; q++) {
    // Grab first point for the first dimension
    for (int point = 0; point < limit; point++) {
      value = data[0][limit * q + point];
      if (value == UINT_MAX) {
        break;
      }
      // printf("Query point: %d, First point: %u\n", q, value);
      for (int dim = 1; dim < numDims / 3; dim++) {
        // printf("\tDim: %d\n", dim);
        // For all points in the next dim
        for (int p = 0; p < limit; p++) {
          if ( data[dim][limit * q + p] == UINT_MAX) {
            break;
          }
          // printf("\t\tPoint: %d, value: %u\n", p, data[dim][limit * q + p]);
          if (value == data[dim][limit * q + p])
          check = 1;
        }
        if (!check) {
          break;
        }
      }
      if (check) {
        result[limit * q + write_index] = value;
        write_index++;
      }
    }
    write_index = 0;
  }
}

// Add batching
double calcDistSums(RTNNState state, unsigned int *check) {
  double *sums;
  float3 *d_points;
  float3 *d_queries;
  unsigned int *d_check;
  float3 *flattenedPoints = (float3 *) malloc( state.numPoints * state.dim * sizeof(float3));
  float3 *flattenedQueries = (float3 *) malloc( state.numQueries * state.dim * sizeof(float3));
  flattenArray(state.h_ndpoints, flattenedPoints, state.dim / 3, state.numPoints);
  flattenArray(state.h_ndqueries, flattenedQueries, state.dim / 3, state.numQueries);

  cudaMalloc((void **) &d_points, state.numPoints * state.dim * sizeof(float3));
  cudaMalloc((void **) &d_queries, state.numQueries * state.dim * sizeof(float3));
  cudaMalloc((void **) &d_check, state.numQueries * state.params.limit * sizeof(unsigned int));
  cudaMemcpy(d_check, check, state.numQueries * state.params.limit * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points, flattenedPoints,state.numPoints * state.dim * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, flattenedQueries, state.numQueries * state.dim * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMallocManaged((void **) &sums, state.numQueries * state.params.limit * sizeof(double));

  findClosestPoints<<<state.numQueries, state.params.limit>>>(d_points, d_queries, d_check, sums, state.radius, state.dim, state.numPoints, state.numQueries, state.params.limit);
  cudaDeviceSynchronize();

  double totalSum = 0;
  for (int i = 0; i < state.numQueries * state.params.limit; i++) {
    totalSum += sums[i];
  }

  cudaFree(d_points);
  cudaFree(d_queries);
  cudaFree(d_check);
  cudaFree(sums);

  return totalSum;
}

// TODO: Split work by blocks
double calcDistSumsBruteForce(RTNNState state) {
  double *sums;
  float3 *d_points;
  float3 *d_queries;
  float3 *flattenedPoints = (float3 *) malloc( state.numPoints * state.dim * sizeof(float3));
  float3 *flattenedQueries = (float3 *) malloc( state.numQueries * state.dim * sizeof(float3));
  flattenArray(state.h_ndpoints, flattenedPoints, state.dim / 3, state.numPoints);
  flattenArray(state.h_ndqueries, flattenedQueries, state.dim / 3, state.numQueries);

  cudaMalloc((void **) &d_points, state.numPoints * state.dim * sizeof(float3));
  cudaMalloc((void **) &d_queries, state.numQueries * state.dim * sizeof(float3));
  cudaMemcpy(d_points, flattenedPoints,state.numPoints * state.dim * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_queries, flattenedQueries, state.numQueries * state.dim * sizeof(float3), cudaMemcpyHostToDevice);
  cudaMallocManaged((void **) &sums, state.numQueries * state.numPoints * sizeof(double));

  findClosestPointsBruteForce<<<state.numQueries, state.numPoints>>>(d_points, d_queries, sums, state.radius, state.dim, state.numPoints, state.numQueries, state.params.limit);
  cudaDeviceSynchronize();

  double totalSum = 0;
  for (int i = 0; i < state.numQueries * state.numPoints; i++) {
    totalSum += sums[i];
  }


  cudaFree(d_points);
  cudaFree(d_queries);
  cudaFree(sums);
  return totalSum;
}

__global__ void findClosestPoints(float3 *points, float3 *queries, unsigned int *check, double *sums, const double epsilon, const int numDims, const int numPoints, const int numQueries, const int limit) {
	unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  
	if (threadIdx.x >= limit) {
		return;
	}

  // Max dimensions is 15 for now.
	float3 query[5];
  for (int dim = 0; dim < numDims / 3; dim++) {
    query[dim] = queries[get2DIndex(dim, blockIdx.x , numQueries)];
  }
  
  float3 point;
  double sum = 0;
  int index;
  index = check[get2DIndex(blockIdx.x, threadIdx.x, limit)];
  if (index == UINT_MAX) {
    return;
  }

  for (int dim = 0; dim < numDims / 3; dim++) {
    point = points[get2DIndex(dim, index, numPoints)];
    if (point.x == query[dim].x && point.y == query[dim].y && point.z == query[dim].z) {
      continue;
    }
    // printf("Query %u: [%f, %f, %f], Point %u: [%f, %f, %f], distance: %f\n", tid, query[dim].x, query[dim].y, query[dim].z, i, point.x, point.y, point.z, dot(query[dim] - point, query[dim] - point));
    sum += sqr(point.x - query[dim].x);
    sum += sqr(point.y - query[dim].y);
    sum += sqr(point.z - query[dim].z);
  }

  double distance = sqrt(sum);
  if (distance <= epsilon) {
    sums[tid] += distance;
  }
}

__global__ void findClosestPointsBruteForce(float3 *points, float3 *queries, double *sums, const double epsilon, const int numDims, const int numPoints, const int numQueries, const int limit) {
  unsigned int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  
	if (threadIdx.x >= numPoints) {
		return;
	}

  // Max dimensions is 15 for now.
	float3 query[5];
  for (int dim = 0; dim < numDims / 3; dim++) {
    query[dim] = queries[get2DIndex(dim, blockIdx.x , numQueries)];
  }
  
  float3 point;
  double distance = 0;

  for (int dim = 0; dim < numDims / 3; dim++) {
    point = points[get2DIndex(dim, threadIdx.x, numPoints)];
    if (point.x == query[dim].x && point.y == query[dim].y && point.z == query[dim].z) {
      continue;
    }
    // printf("Query %u: [%f, %f, %f], Point %u: [%f, %f, %f], distance: %f\n", tid, query[dim].x, query[dim].y, query[dim].z, i, point.x, point.y, point.z, dot(query[dim] - point, query[dim] - point));
    distance += sqr(point.x - query[dim].x);
    distance += sqr(point.y - query[dim].y);
    distance += sqr(point.z - query[dim].z);
  }

  if (distance < epsilon * epsilon) {
    // printf("Distance: %f\n", sqrt(distance));
    sums[tid] += sqrt(distance);
  }
}

// Does not work when the points and queries are sorted.
__device__ int isIn(unsigned int *in, unsigned int value, size_t in_size) {
  int check = 0;
  for (int i = 0; i < in_size; i++) {
    if (in[i] == value) {
      check = 1;
      break;
    }
  }
  return check;
}

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

double sumDistances(RTNNState state, unsigned int **check) {
  int index = 0;
  double *totalSum = (double *) malloc(state.numQueries * sizeof(double));
  double totalDist = 0;
  
  for (int b = 0; b < state.numOfBatches; b++) {
    for (int q = 0; q < state.numQueries; q++) {
      totalSum[q] = 0;
      for (int p = 0; p < state.numPoints; p++) {
        for (int d = 0; d < state.dim / 3; d++) {
          // index = check[b][get2DIndex(q, p, state.params.limit)];
          // if (index == UINT_MAX) {
          //   continue;
          // }
          if (isnan(state.distances[d][get3DIndex(b, q, p, state.numQueries, state.numPoints)])) {
            continue;
          }
          totalSum[q] += state.distances[d][get3DIndex(b, q, p, state.numQueries, state.numPoints)];
          // printf("Query %u, Point %u, Index, %u, Squared distance: %f\n", q, p, index, state.distances[d][get3DIndex(b, q, index, state.numQueries, state.numPoints)]);
        }
      }
      if (totalSum[q] < state.radius * state.radius) {
        // printf("Distance: %f\n", sqrt(totalSum[q]));
        totalDist += sqrt(totalSum[q]);
      }
    }
  }
  return totalDist;
}
#include <sutil/Exception.h>
#include <sutil/Timing.h>
#include <thrust/device_vector.h>

#include "optixNSearch.h"
#include "state.h"
#include "func.h"

void splitRays(RTNNState& state, thrust::device_ptr<float3> d_actQs, uint64_t batch_id, uint64_t raysPerSplit, uint64_t splitNum) {
  // TODO: extra rays
  // printf("Rays per split: %d, split num: %d, batch id: %d\n", raysPerSplit, splitNum, batch_id);
  thrust::copy(&state.d_actQs[batch_id][splitNum * raysPerSplit], &state.d_actQs[batch_id][splitNum * raysPerSplit + raysPerSplit], d_actQs);

    // Print queries for testing purposes
    // float3 *h_actQs = new float3[raysPerSplit];
    // thrust::copy(d_actQs, d_actQs+raysPerSplit, h_actQs);

    // for(int i=0; i<raysPerSplit; i++)
    //     std::cout << h_actQs[i].x << " " << h_actQs[i].y << " " << h_actQs[i].z << std::endl;
}

void search(RTNNState& state, int batch_id) {
  printf("Available mem: %f\n", calcMemUsage(state));
  Timing::startTiming("batch search");
    
    Timing::startTiming("\tallocate memory");
      unsigned int numSplit = 10;
      uint64_t numQueries = state.numActQueries[batch_id] / numSplit;
      thrust::device_ptr<float3> d_actQs;
      allocThrustDevicePtr(&d_actQs, numQueries, &state.d_pointers);

      state.params.limit = state.knn;
      thrust::device_ptr<unsigned int> output_buffer;
      // TODO: Change to uint64_t;
      allocThrustDevicePtr(&output_buffer, numQueries * state.params.limit, &state.d_pointers);

      // Store iterator as uint64_t

      thrust::device_ptr<double> dist_buffer;
      state.params.distances = allocThrustDevicePtr(&dist_buffer, state.numActQueries[batch_id], &state.d_pointers);
      // thrust::fill(dist_buffer, dist_buffer + state.numQueries, 0.0);

      //TODO: K: Break up copy
      unsigned int *data;
      cudaMallocHost(reinterpret_cast<void**>(&data), state.numActQueries[batch_id] * state.params.limit * (uint64_t) sizeof(unsigned int));
      // data = (unsigned int *) malloc(state.numActQueries[batch_id] * state.params.limit * sizeof(unsigned int));
      state.h_res[batch_id] = data;

      double *distances;
      distances = (double *) malloc(state.numActQueries[batch_id] * sizeof(double));
    Timing::stopTiming(true);

    for (uint64_t i = 0; i < numSplit; i++) {
      Timing::startTiming("\tpre-search computation");
        // unused slots will become UINT_MAX
        fillByValue(output_buffer, (uint64_t) numQueries * state.params.limit, UINT_MAX);
        splitRays(state, d_actQs, batch_id, numQueries, i);

        if (state.qGasSortMode && !state.toGather) state.params.d_r2q_map = state.d_r2q_map[batch_id];
        else state.params.d_r2q_map = nullptr; // if no GAS-sorting or has done gather, this map is null.

        state.params.mode = PRECISE;
        if ((state.searchMode == "radius") && state.partition && (batch_id < state.numOfBatches - 1)) {
          // note that hardware AABB test during traversal in the current OptiX
          // implementation is inherently approximate, so if we want to guarantee
          // that a point is inside an AABB we still have to do an explicit aabb
          // test (instead of using NOTEST). see:
          // https://forums.developer.nvidia.com/t/numerical-imprecision-in-intersection-test/183665/4.

          // in radius mode use AABBTEST except for the last batch. see how the
          // launchRadius is calculated in the |genBatches| function. AABBTEST is
          // faster than PRECISE since sphere test is much more costly then aabb test.
          state.params.mode = AABBTEST;
        }

        state.params.radius = state.launchRadius[batch_id];
      Timing::stopTiming(true);
      
      Timing::startTiming("\toptix search");
        launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id, numQueries, thrust::raw_pointer_cast(d_actQs));
        cudaDeviceSynchronize();
        OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
      Timing::stopTiming(true);

      Timing::startTiming("\tresult copy D2H");
      
        CUDA_CHECK( cudaMemcpy(
                        &data[(i * numQueries * state.params.limit)],
                        thrust::raw_pointer_cast(output_buffer),
                        numQueries * state.params.limit * (uint64_t) sizeof(unsigned int),
                        cudaMemcpyDeviceToHost
                        // state.stream[batch_id]
                        ) );
      // OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
      cudaDeviceSynchronize();
      Timing::stopTiming(true);

      Timing::startTiming("\tdistances copy D2H");
        CUDA_CHECK( cudaMemcpyAsync(
                        distances,
                        thrust::raw_pointer_cast(state.params.distances),
                        state.numActQueries[batch_id] * sizeof(double),
                        cudaMemcpyDeviceToHost,
                        state.stream[batch_id]
                        ) );
        OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
      Timing::stopTiming(true);
    }

    // TODO: K: do with thrust
    // double d_distSum = thrust::reduce(dist_buffer, dist_buffer + state.numActQueries[batch_id]);
    double h_distSum = 0;
    // // thrust::copy(d_distSum, d_distSum + 1, &h_distSum);
    for (uint64_t i = 0; i < state.numActQueries[batch_id]; i++) {
      h_distSum += distances[i];
    }
    printf("New distance sum: %f\n", h_distSum);

    Timing::startTiming("\tfree memory");
      cudaFree(thrust::raw_pointer_cast(d_actQs));
      state.d_pointers.erase(state.d_pointers.find(thrust::raw_pointer_cast(d_actQs)));

      cudaFree(thrust::raw_pointer_cast(output_buffer));
      state.d_pointers.erase(state.d_pointers.find(thrust::raw_pointer_cast(output_buffer)));
    Timing::stopTiming(true);
    
  Timing::stopTiming(true);

  // this frees device memory but will block until the previous optix launch finish and the res is written back.
  //CUDA_CHECK( cudaFree( (void*)thrust::raw_pointer_cast(output_buffer) ) );
}

thrust::device_ptr<unsigned int> initialTraversal(RTNNState& state, int batch_id) {
  Timing::startTiming("initial traversal");
    unsigned int numQueries = state.numActQueries[batch_id];

    state.params.limit = 1;
    thrust::device_ptr<unsigned int> output_buffer;
    allocThrustDevicePtr(&output_buffer, numQueries * state.params.limit, &state.d_pointers);
    // for initial sort fill with 0. it's possible that a query has no
    // neighbors (no intersection with any of the AABB), in which case during
    // gas-sort using FHCoord, gather might use UINT_MAX as a key if filled
    // with UINT_MAX.
    fillByValue(output_buffer, numQueries * state.params.limit, 0);

    state.params.d_r2q_map = nullptr; // contains the index to reorder rays
    state.params.mode = NOTEST;
    state.params.radius = state.launchRadius[batch_id]; // doesn't quite matter since we never check radius in approx mode

    launchSubframe( thrust::raw_pointer_cast(output_buffer), state, batch_id);
    // TODO: could delay this until sort, but initial traversal is lightweight anyways
    OMIT_ON_E2EMSR( CUDA_CHECK( cudaStreamSynchronize( state.stream[batch_id] ) ) );
  Timing::stopTiming(true);

  return output_buffer;
}

void gasSortSearch(RTNNState& state, int batch_id) {
  // TODO: maybe we should have a third mode where we sort FH primitives in
  // z-order or raster order. This would improve the performance when no
  // pre-sorting is done, and might even out-perform it since we are sorting
  // fewer points.

  // Initial traversal to aggregate the queries
  thrust::device_ptr<unsigned int> d_firsthit_idx_ptr = initialTraversal(state, batch_id);

  // Generate the GAS-sorted query order
  thrust::device_ptr<unsigned int> d_indices_ptr;
  if (state.qGasSortMode == 1)
    d_indices_ptr = sortQueriesByFHCoord(state, d_firsthit_idx_ptr, batch_id);
  else if (state.qGasSortMode == 2)
    d_indices_ptr = sortQueriesByFHIdx(state, d_firsthit_idx_ptr, batch_id);

  // Actually sort queries in memory if toGather is enabled
  if (state.toGather)
    gatherQueries( state, d_indices_ptr, batch_id );
}

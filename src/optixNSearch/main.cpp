#include <sutil/Exception.h>
#include <sutil/Timing.h>

#include "func.h"
#include "grid.h"
#include "optixNSearch.h"
#include "state.h"

void setDevice(RTNNState &state) {
  int32_t device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, state.device_id));
  CUDA_CHECK(cudaSetDevice(state.device_id));
  std::cerr << "\tUsing [" << state.device_id << "]: " << prop.name
            << std::endl;
  state.totDRAMSize = (double)prop.totalGlobalMem / 1024 / 1024 / 1024;
  std::cerr << "\tMemory: " << state.totDRAMSize << " GB" << std::endl;
  // conservatively reduce dram size by 256 MB as the usable memory appears to
  // be that much smaller than what is reported, presumably to store data
  // structures that are hidden from us.
  state.totDRAMSize -= 0.25;
}

void freeGridPointers(RTNNState &state) {
  for (auto it = state.d_gridPointers.begin(); it != state.d_gridPointers.end();
       it++) {
    CUDA_CHECK(cudaFree(*it));
  }
  // fprintf(stdout, "Finish early free\n");
}

void setupSearch(RTNNState &state) {
  if (!state.deferFree)
    freeGridPointers(state);

  if (state.partition)
    return;

  // assert(state.numOfBatches == -1);
  state.numOfBatches = 1;

  state.numActQueries[0] = state.numQueries;
  state.d_actQs[0] = state.params.queries;
  state.h_actQs[0] = state.h_queries;
  state.launchRadius[0] = state.radius;
}

int main(int argc, char *argv[]) {
  RTNNState state;

  parseArgs(state, argc, argv);

  // Read and store points
  readDataByDim(state);

  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "Deferred free? " << std::boolalpha << state.deferFree
            << std::endl;
  std::cout << "E2E Measure? " << std::boolalpha << state.msr << std::endl;
  std::cout << "K: " << state.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Query partition? " << std::boolalpha << state.partition
            << std::endl;
  std::cout << "Approx query partition mode: " << state.approxMode << std::endl;
  std::cout << "Auto batching? " << std::boolalpha << state.autoNB << std::endl;
  std::cout << "Auto crRatio? " << std::boolalpha << state.autoCR << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio
            << std::endl; // only useful when preSort == 1/2 and autoCR is false
  std::cout << "mcScale: " << state.mcScale << std::endl;
  std::cout << "crStep: " << state.crStep << std::endl;
  std::cout << "Interleave? " << std::boolalpha << state.interleave
            << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << state.querySortMode << std::endl;
  std::cout << "gsrRatio: " << state.gsrRatio
            << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather after gas sort? " << std::boolalpha << state.toGather
            << std::endl;
  std::cout << "========================================" << std::endl
            << std::endl;

  try {
    setDevice(state);

    Timing::reset();
    
    state.h_queries = (float3*)malloc(state.numQueries * sizeof(float3));
    state.h_points = (float3*)malloc(state.numPoints * sizeof(float3));
    thrust::device_ptr<float3> d_points_ptr;
    thrust::device_ptr<float3> d_queries_ptr;
    allocateData(state, &d_points_ptr, &d_queries_ptr);

    // Create result array that stores the point ID of all points epsilon away by dimension.
    unsigned int **result_prims_by_batch = (unsigned int **) malloc(state.numOfBatches * sizeof(unsigned int *));
    for (int b = 0; b < state.numOfBatches; b++) {
      result_prims_by_batch[b] = (unsigned int *) malloc(state.numQueries * state.params.limit * sizeof(unsigned int));
    }

    for (int dim = 0; dim < state.dim / 3; dim++) {
      state.currentDim = dim;

      setPointsByDim(state, dim);
      // printf("----------\nInput Data\n");
      // for (int p = 0; p < state.numPoints; p++) {
      //   std::cout << state.h_points[p].x << ", " << state.h_points[p].y << ", " << state.h_points[p].z << std::endl;
      // }
      // printf("----------\n");
      // for (int q = 0; q < state.numQueries; q++) {
      //   std::cout << state.h_queries[q].x << ", " << state.h_queries[q].y << ", " << state.h_queries[q].z << std::endl;
      // }
      // printf("----------\n");

      // Copy points/queries to device
      uploadData(state, &d_points_ptr, &d_queries_ptr);

      // call this after set device.
      // Create CUDA streams
      // TODO: K: move outside of loop
      initBatches(state);

      // Create context, pipeline, and SBT
      // TODO: K: move outside of loop
      setupOptiX(state);

      Timing::startTiming("total search time");

      // TODO: streamline the logic of partition and sorting.
      sortParticles(state, QUERY, state.querySortMode);

      // samepq indicates same underlying data and sorting mode, in which case
      // queries have been sorted so no need to sort them again.
      if (!state.samepq)
        sortParticles(state, POINT, state.pointSortMode);

      // early free done here too
      setupSearch(state);

      if (state.interleave) {
        for (int i = 0; i < state.numOfBatches; i++) {
          // it's possible that certain batches have 0 query (e.g., state.partThd
          // too low).
          if (state.numActQueries[i] == 0)
            continue;
          // TODO: group buildGas together to allow overlapping; this would allow
          // us to batch-free temp storages and non-compacted gas storages. right
          // now free storage serializes gas building.
          createGeometry(
              state, i,
              state.launchRadius[i] /
                  state.gsrRatio); // batch_id ignored if not partition.
        }

        for (int i = 0; i < state.numOfBatches; i++) {
          if (state.numActQueries[i] == 0)
            continue;
          if (state.qGasSortMode)
            gasSortSearch(state, i);
        }

        for (int i = 0; i < state.numOfBatches; i++) {
          if (state.numActQueries[i] == 0)
            continue;
          if (state.qGasSortMode && state.gsrRatio != 1)
            createGeometry(state, i, state.launchRadius[i]);
        }

        for (int i = 0; i < state.numOfBatches; i++) {
          if (state.numActQueries[i] == 0)
            continue;
          // TODO: when K is too big, we can't launch all rays together. split
          // rays.
          search(state, i);
        }
      } else {
        for (int i = 0; i < state.numOfBatches; i++) {
          if (state.numActQueries[i] == 0)
            continue;

          // create the GAS using the current order of points and the launchRadius
          // of the current batch.
          // TODO: does it make sense to have per-batch |gsrRatio|?
          createGeometry(
              state, i,
              state.launchRadius[i] /
                  state.gsrRatio); // batch_id ignored if not partition.

          if (state.qGasSortMode) {
            gasSortSearch(state, i);
            if (state.gsrRatio != 1)
              createGeometry(state, i, state.launchRadius[i]);
          }

          search(state, i);
        }
      }

      if (state.sanCheck)
        sanityCheck(state);

      // Store results in a result array
      // for (int b = 0; b < state.numOfBatches; b++) {
      //   if (state.numActQueries[b] == 0)
      //     continue;
      //   result_prims_by_batch[b] = reinterpret_cast<unsigned int*>( state.h_res[b]);
      // }
    }

    Timing::startTiming("calculate distance sums");
    printf("Total Distance: %f\n", sumDistances(state, result_prims_by_batch));
    Timing::stopTiming(true);
    
    // // Intersection of results array
    // unsigned int *result_array = (unsigned int *) malloc(state.numQueries * state.params.limit * sizeof(unsigned int));
    // memset(result_array, UINT_MAX, state.numQueries * state.params.limit * sizeof(unsigned int));
    // Timing::startTiming("calculate intersections of neighboring points");
    // calcIntersection(result_prims_by_dim, result_array, state.dim, state.numQueries, state.params.limit);
    // Timing::stopTiming(true);

    // // printf("Resulting points: \n");
    // // for (int i = 0; i < state.numQueries; i++) {
    // //   printf("Query point: %u\n", i);
    // //   for (int j = 0; j < state.params.limit; j++) {
    // //     if (result_array[i * state.params.limit + j] == UINT_MAX) {
    // //       break;
    // //     }
    // //     printf("\tPoint: %u\n", result_array[i * state.params.limit + j]);
    // //   }
    // // }

    // // CUDA-based distance calculations
    // Timing::startTiming("calculate distance sums");
    // double totalSum = calcDistSums(state, result_array);
    // Timing::stopTiming(true);

    CUDA_SYNC_CHECK();
    Timing::stopTiming(true);

    Timing::startTiming("calculate distance sums brute force");
    double totalSum = calcDistSumsBruteForce(state);
    Timing::stopTiming(true);

    // printf("\nTotal Distance Sum: %f\n", totalSum);
    printf("Total Distance Sum Brute Force: %f\n", totalSum);

      
    cleanupState(state);
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
    exit(1);
  }

  exit(0);
}

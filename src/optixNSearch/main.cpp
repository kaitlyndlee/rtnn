#include <sutil/Exception.h>
#include <sutil/Timing.h>

#include "optixNSearch.h"
#include "state.h"
#include "func.h"
#include "grid.h"
#include <omp.h>

// void bruteForceSearch(RTNNState state, int batch) {
//   double *totalSum = (double *) malloc(state.numActQueries[batch] * sizeof(double));
//   #pragma omp parallel for
//   for (unsigned int q = 0; q < state.numActQueries[batch]; q++) {
//     totalSum[q] = 0;
//     double distance = 0;
//     float3 query = state.h_queries[q];
//     for (unsigned int p = 0; p < state.numPoints; p++) {
//       float3 point = state.h_points[p];
//       distance += sqr(point.x - query.x);
//       distance += sqr(point.y - query.y);
//       distance += sqr(point.z - query.z);
//       if (distance < state.radius * state.radius) {
//         totalSum[q] += distance;
//       }
//     }
//   }
// }

void setDevice ( RTNNState& state ) {
  int32_t device_count = 0;
  CUDA_CHECK( cudaGetDeviceCount( &device_count ) );
  std::cerr << "\tTotal GPUs visible: " << device_count << std::endl;
  
  cudaDeviceProp prop;
  CUDA_CHECK( cudaGetDeviceProperties ( &prop, state.device_id ) );
  CUDA_CHECK( cudaSetDevice( state.device_id ) );
  std::cerr << "\tUsing [" << state.device_id << "]: " << prop.name << std::endl;
  state.totDRAMSize = (double)prop.totalGlobalMem/1024/1024/1024;
  std::cerr << "\tMemory: " << state.totDRAMSize << " GB" << std::endl;
  // conservatively reduce dram size by 256 MB as the usable memory appears to
  // be that much smaller than what is reported, presumably to store data
  // structures that are hidden from us.
  state.totDRAMSize -= 0.25;

  warm_up_gpu(state.device_id);
}

void freeGridPointers( RTNNState& state ) {
  for (auto it = state.d_gridPointers.begin(); it != state.d_gridPointers.end(); it++) {
    CUDA_CHECK( cudaFree( *it ) );
  }
  //fprintf(stdout, "Finish early free\n");
}

void setupSearch( RTNNState& state ) {
  if (!state.deferFree) freeGridPointers(state);

  if (state.partition) return;

  // assert(state.numOfBatches == -1);
  state.numOfBatches = 1;

  state.numActQueries[0] = state.numQueries;
  state.d_actQs[0] = state.params.queries;
  state.h_actQs[0] = state.h_queries;
  state.launchRadius[0] = state.radius;
}

void startSearch(RTNNState& state) {
  if (state.interleave) {
    for (int i = 0; i < state.numOfBatches; i++) {
      // it's possible that certain batches have 0 query (e.g., state.partThd too low).
      if (state.numActQueries[i] == 0) continue;
    // TODO: group buildGas together to allow overlapping; this would allow
    // us to batch-free temp storages and non-compacted gas storages. right
    // now free storage serializes gas building.
      createGeometry (state, i, state.launchRadius[i]/state.gsrRatio); // batch_id ignored if not partition.
    }

    for (int i = 0; i < state.numOfBatches; i++) {
      if (state.numActQueries[i] == 0) continue;
      if (state.qGasSortMode) gasSortSearch(state, i);
    }

    for (int i = 0; i < state.numOfBatches; i++) {
      if (state.numActQueries[i] == 0) continue;
      if (state.qGasSortMode && state.gsrRatio != 1)
        createGeometry (state, i, state.launchRadius[i]);
    }

    for (int i = 0; i < state.numOfBatches; i++) {
      if (state.numActQueries[i] == 0) continue;
      // TODO: when K is too big, we can't launch all rays together. split rays.
      search(state, i);
    }
  } else {
    for (int i = 0; i < state.numOfBatches; i++) {
      if (state.numActQueries[i] == 0) continue;

      // create the GAS using the current order of points and the launchRadius of the current batch.
      // TODO: does it make sense to have per-batch |gsrRatio|?
      createGeometry (state, i, state.launchRadius[i]/state.gsrRatio); // batch_id ignored if not partition.

      if (state.qGasSortMode) {
        gasSortSearch(state, i);
        if (state.gsrRatio != 1)
          createGeometry (state, i, state.launchRadius[i]);
      }

      search(state, i);
    }
  }
}

int main( int argc, char* argv[] )
{
  RTNNState state;
  parseArgs( state, argc, argv );

  Timing::startTiming("read the whole data");
    readDataByDim(state);
  Timing::stopTiming(true);


  std::cout << "========================================" << std::endl;
  std::cout << "numPoints: " << state.numPoints << std::endl;
  std::cout << "numQueries: " << state.numQueries << std::endl;
  std::cout << "searchMode: " << state.searchMode << std::endl;
  std::cout << "radius: " << state.radius << std::endl;
  std::cout << "Deferred free? " << std::boolalpha << state.deferFree << std::endl;
  std::cout << "E2E Measure? " << std::boolalpha << state.msr << std::endl;
  std::cout << "K: " << state.knn << std::endl;
  std::cout << "Same P and Q? " << std::boolalpha << state.samepq << std::endl;
  std::cout << "Query partition? " << std::boolalpha << state.partition << std::endl;
  std::cout << "Approx query partition mode: " << state.approxMode << std::endl;
  std::cout << "Auto batching? " << std::boolalpha << state.autoNB << std::endl;
  std::cout << "Auto crRatio? " << std::boolalpha << state.autoCR << std::endl;
  std::cout << "cellRadiusRatio: " << std::boolalpha << state.crRatio << std::endl; // only useful when preSort == 1/2 and autoCR is false
  std::cout << "mcScale: " << state.mcScale << std::endl;
  std::cout << "crStep: " << state.crStep << std::endl;
  std::cout << "Interleave? " << std::boolalpha << state.interleave << std::endl;
  std::cout << "qGasSortMode: " << state.qGasSortMode << std::endl;
  std::cout << "pointSortMode: " << state.pointSortMode << std::endl;
  std::cout << "querySortMode: " << state.querySortMode << std::endl;
  std::cout << "gsrRatio: " << state.gsrRatio << std::endl; // only useful when qGasSortMode != 0
  std::cout << "Gather after gas sort? " << std::boolalpha << state.toGather << std::endl;
  std::cout << "========================================" << std::endl << std::endl;

  try
  {
    setDevice(state);
    Timing::reset();
    
    // Allocate data once and reuse the memory for every third dimension
    Timing::startTiming("allocate points and queries device pointers");
      thrust::device_ptr<float3> d_points_ptr;
      thrust::device_ptr<float3> d_queries_ptr;
      allocateData(state, &d_points_ptr, &d_queries_ptr);
    Timing::stopTiming(true);

    Timing::startTiming("setup optix and batching");
      initBatches(state);
      setupOptiX(state);
    Timing::stopTiming(true);

    Timing::startTiming("total search");
    char str[100];
    for (int dim = 0; dim < state.dim / 3; dim++) {
      printf("========================================\n");
      sprintf(str, "dim %d search", dim);
      Timing::startTiming(str);
        Timing::startTiming("\tcopy points/queries to device");
          state.currentDim = dim;
          setPointsByDim(state, dim);
          uploadData(state, &d_points_ptr, &d_queries_ptr);
        Timing::stopTiming(true);

          // printf("----------\nInput Data\n");
          // for (int p = 0; p < state.numPoints; p++) {
          //   std::cout << state.h_points[p].x << ", " << state.h_points[p].y << ", " << state.h_points[p].z << std::endl;
          // }
          // printf("----------\n");
          // for (int q = 0; q < state.numQueries; q++) {
          //   std::cout << state.h_queries[q].x << ", " << state.h_queries[q].y << ", " << state.h_queries[q].z << std::endl;
          // }
          // printf("----------\n");

          // calcMemoryUsage(state);

        Timing::startTiming("\tsort points and queries");
          // TODO: streamline the logic of partition and sorting.
          sortParticles(state, QUERY, state.querySortMode);

          // samepq indicates same underlying data and sorting mode, in which case
          // queries have been sorted so no need to sort them again.
          if (!state.samepq) sortParticles(state, POINT, state.pointSortMode);
        Timing::stopTiming(true);
        
        Timing::startTiming("\tsetup search");
          // early free done here too
          setupSearch(state);
        Timing::stopTiming(true);

        startSearch(state);
      Timing::stopTiming(true);

      if(state.sanCheck) sanityCheck(state);
    }
      
    CUDA_SYNC_CHECK();
    Timing::stopTiming(true);

    Timing::startTiming("brute force search time");
      bruteForceSearch(state.h_points, state.h_queries, state.radius, state.numPoints, state.numQueries, state.params.limit);
    Timing::stopTiming(true);

    Timing::startTiming("cleanup the state");
      cleanupState(state);
    Timing::stopTiming(true);
  }
  catch( std::exception& e )
  {
    std::cerr << "Caught exception: " << e.what() << "\n";
    exit(1);
  }

  exit(0);
}

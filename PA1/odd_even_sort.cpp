#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "worker.h"

void Worker::sort() {
  if (out_of_range || block_len == 0) {
    return;
  }
  std::sort(data, data + block_len);
  size_t b_size = (n + nprocs - 1) / nprocs;
  int active_procs = (n + b_size - 1) / b_size;
  float* recv_data = new float[b_size];
  float* temp = new float[b_size];
  for (int i = 0; i < nprocs; ++i) {
    int local_changed = 0;
    int neighbor = (i & 1) ? (rank & 1 ? rank + 1 : rank - 1) : (rank & 1 ? rank - 1 : rank + 1);
    if (neighbor >= 0 && neighbor < active_procs) {
      size_t neighbor_len = (neighbor == active_procs - 1) ? (n - neighbor * b_size) : b_size;
      bool need_exchange = 0;
      if (rank < neighbor) {
        float neighbor_min;
        MPI_Sendrecv(&data[block_len - 1], 1, MPI_FLOAT, neighbor, 0, 
                     &neighbor_min, 1, MPI_FLOAT, neighbor, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (data[block_len - 1] > neighbor_min) {
          need_exchange = 1;
        }
      } else {
        float neighbor_max;
        MPI_Sendrecv(&data[0], 1, MPI_FLOAT, neighbor, 0, 
                     &neighbor_max, 1, MPI_FLOAT, neighbor, 0, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (data[0] < neighbor_max) {
          need_exchange = 1;
        }
      }

      if (need_exchange) {
        local_change = 1;
        MPI_Sendrecv(data, block_len, MPI_FLOAT, neighbor, 1, 
                     recv_data, neighbor_len, MPI_FLOAT, neighbor, 1, 
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank < neighbor) {
          size_t i = 0, j = 0, k = 0;
          while (k < block_len) {
            if (j == neighbor_len || (i < block_len && data[i] <= recv_data[j])) {
              temp[k++] = data[i++];
            } else {
              temp[k++] = recv_data[j++];
            }
          }
        } else {
          long long i = (long long)block_len - 1;
          long long j = (long long)neighbor_len - 1;
          long long k = (long long)block_len - 1;
          while (k >= 0) {
            if (j < 0 || (i >= 0 && data[i] >= recv_data[j])) {
              temp[k--] = data[i--];
            } else {
              temp[k--] = recv_data[j--];
            }
          }
        }
        memcpy(data, temp, block_len * sizeof(float));
      }
    }
    int global_changed = 0;
    MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (global_changed == 0) {
      break;
    }
  }
  delete[] recv_data;
  delete[] temp;
}
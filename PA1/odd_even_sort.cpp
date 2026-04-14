#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "worker.h"

static void radix_sort(float* arr, float* temp, size_t len) {
  uint32_t* src = reinterpret_cast<uint32_t*>(arr);
  uint32_t* dst = reinterpret_cast<uint32_t*>(temp);
  for (size_t i = 0; i < len; ++i) {
    uint32_t u = src[i];
    src[i] = (u >> 31) ? ~u : (u | 0x80000000);
  }
  for (int shift = 0; shift < 32; shift += 8) {
    uint32_t count[256] = {0};
    for (size_t i = 0; i < len; ++i) {
      count[(src[i] >> shift) & 0xFF]++;
    }
    uint32_t pos[256];
    pos[0] = 0;
    for (int i = 1; i < 256; ++i) {
        pos[i] = pos[i - 1] + count[i - 1];
    }
    for (size_t i = 0; i < len; ++i) {
      dst[pos[(src[i] >> shift) & 0xFF]++] = src[i];
    }
    uint32_t* tmp = src;
    src = dst;
    dst = tmp;
  }
  for (size_t i = 0; i < len; ++i) {
    uint32_t u = src[i];
    src[i] = (u >> 31) ? (u & 0x7FFFFFFF) : ~u;
  }
}

void Worker::sort() {
  if (out_of_range || block_len == 0) {
    return;
  }
  size_t b_size = (n + nprocs - 1) / nprocs;
  int active_procs = (n + b_size - 1) / b_size;
  float* bufferB = new float[b_size];
  float* recv_data = new float[b_size];
  if (block_len > 10000) {
    radix_sort(data, bufferB, block_len);
  } else {
    std::sort(data, data + block_len);
  }
  float* src = data;
  float* dst = bufferB;
  for (int shift = 0; shift < nprocs; shift++) {
    int neighbor = (shift & 1) ? (rank & 1 ? rank + 1 : rank - 1) : (rank & 1 ? rank - 1 : rank + 1);
    if (neighbor >= 0 && neighbor < active_procs) {
      size_t neighbor_len = (neighbor == active_procs - 1) ? (n - neighbor * b_size) : b_size;
      float my_pivot, recv_pivot;
      my_pivot = (rank < neighbor) ? src[block_len - 1] : src[0];
      MPI_Sendrecv(&my_pivot, 1, MPI_FLOAT, neighbor, 0,
                   &recv_pivot, 1, MPI_FLOAT, neighbor, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      bool need_exchange = ((rank < neighbor) && (my_pivot > recv_pivot)) || ((rank > neighbor) && (my_pivot < recv_pivot));
      if (need_exchange) {
        // ---------------------------------------------------------
        // Phase 1: 通过网络二分查找，精确定位需要交换的元素数量 K
        // ---------------------------------------------------------
        long long low = 0;
        long long high = std::min(block_len, neighbor_len);
        long long K = 0; 
        
        while (low <= high) {
          long long mid = low + (high - low) / 2;

          // 准备向对方发送边界采样值进行校验
          float my_vals[2], recv_vals[2];
          const float NEG_INF = -1e38f;
          const float POS_INF = 1e38f;

          if (rank < neighbor) {
            // 左侧进程：试图给出最大的 mid 个元素
            my_vals[0] = (mid < block_len) ? src[block_len - mid - 1] : NEG_INF; // a_left
            my_vals[1] = (mid > 0) ? src[block_len - mid] : POS_INF;             // a_right
          } else {
            // 右侧进程：试图给出最小的 mid 个元素
            my_vals[0] = (mid > 0) ? src[mid - 1] : NEG_INF;                     // b_left
            my_vals[1] = (mid < block_len) ? src[mid] : POS_INF;                 // b_right
          }

          // 使用 tag=2 进行极小数据量的边界对齐通信
          MPI_Sendrecv(my_vals, 2, MPI_FLOAT, neighbor, 2,
                       recv_vals, 2, MPI_FLOAT, neighbor, 2,
                       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          // 统一人称视角，解析双方的数据
          float a_left, a_right, b_left, b_right;
          if (rank < neighbor) {
            a_left = my_vals[0]; a_right = my_vals[1];
            b_left = recv_vals[0]; b_right = recv_vals[1];
          } else {
            b_left = my_vals[0]; b_right = my_vals[1];
            a_left = recv_vals[0]; a_right = recv_vals[1];
          }

          // 决策逻辑（双方进程拿到的 4 个变量完全一致，因此决策绝对同步）
          if (a_left > b_right) {
            low = mid + 1;  // 左边给出的元素太少了，K 需要增大
          } else if (b_left > a_right) {
            high = mid - 1; // 左边给出的元素太多了，K 需要减小
          } else {
            K = mid;        // 完美找到分割点！
            break;
          }
        }

        // ---------------------------------------------------------
        // Phase 2: 只对这 K 个越界的元素进行精准交换与局部归并
        // ---------------------------------------------------------
        if (K > 0) {
          float* kept_data = nullptr;
          size_t kept_len = block_len - K;

          if (rank < neighbor) {
            // 左侧进程：发送末尾的 K 个大元素，接收对方开头的 K 个小元素
            MPI_Sendrecv(src + kept_len, K, MPI_FLOAT, neighbor, 1,
                         recv_data, K, MPI_FLOAT, neighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            kept_data = src; 
          } else {
            // 右侧进程：发送开头的 K 个小元素，接收对方末尾的 K 个大元素
            MPI_Sendrecv(src, K, MPI_FLOAT, neighbor, 1,
                         recv_data, K, MPI_FLOAT, neighbor, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            kept_data = src + K;
          }

          // Phase 3: 归并过程史诗级大统一！
          // 双方现在手里都有绝对属于自己的 block_len 个元素：
          // 其中 kept_len 个在 kept_data 里（已有序），K 个在 recv_data 里（已有序）。
          // 直接进行标准的正向双指针合并即可！

          size_t i = 0, j = 0, k = 0;

          // 继承我们之前的优化：Skip Merge（二分跳跃）
          if (kept_len > 0) {
            size_t skip_i = std::upper_bound(kept_data, kept_data + kept_len, recv_data[0]) - kept_data;
            if (skip_i > 0) {
              memcpy(dst, kept_data, skip_i * sizeof(float));
              i = skip_i;
              k = skip_i;
            }
          }

          // 无分支归并 (Branchless Merge)
          while (i < kept_len && j < K) {
            bool cmp = kept_data[i] <= recv_data[j];
            dst[k] = cmp ? kept_data[i] : recv_data[j];
            i += cmp;
            j += !cmp;
            k++;
          }

          // 处理剩余尾部
          while (i < kept_len) dst[k++] = kept_data[i++];
          while (j < K) dst[k++] = recv_data[j++];

          // 交换指针，归并完成
          std::swap(src, dst);
        }
      }
    }
  }
  if (src != data) {
    memcpy(data, src, block_len * sizeof(float));
  }
  delete[] recv_data;
  delete[] bufferB;
}

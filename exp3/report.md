# openmp schedule调度性能实验

王宇康 2024010091

## 线程数
```shell
#!/bin/bash
OMP_NUM_THREADS=28 srun -N 1 ./omp_sched
```

使用了 28 个线程

## 指导语句

```C++
#pragma omp parallel for schedule(static)
```

其中 stctic 的分别替换为 dynamic 和 guided 进行相同测试

## 测量结果
static

```
Sort uniform parts: 68.9073 ms
Sort random parts: 189.848 ms
```

dynamic

```
Sort uniform parts: 81.9105 ms
Sort random parts: 166.966 ms
```

guided

```
Sort uniform parts: 69.3632 ms
Sort random parts: 162.938 ms
```

## 原因分析

对于 uniform 任务而言 static 约等于 guided 小于 dynamic  

每个任务的工作量完全一致，static 平分任务量，无需任何同步，而 dynamic 频繁的调度开销超过并行带来的收益，guided 由于在前期完成大部分的分配，因此同步次数小于 dynamic，从而时间接近 static

对于 random 任务而言 guided 约等于 dynamic 小于 static  

由于每个任务任务量极其不平均，使用 dynamic 能有效避免一个线程处理超长数据，而其他线程等待的情况，大大提高并行度，从而表现较好，guided 的分配介于 ddynamic 和 static 之间  

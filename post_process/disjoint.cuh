#include <cuda.h>
#include <cuda_runtime.h>

__device__ void myswap (int& a, int& b) {
  int tmp = a;
  a = b;
  b = tmp;
}
__device__ int FIND (int* L, int index) {
  int label = L[index];
  while (label - 1 != index) {
      index = label - 1;
      label = L[index];
  }
  return index;
}

__device__ void UNION (int* L, int a, int b) {
  bool done = false;
  
  while (done == false) {
      a = FIND(L, a);
      b = FIND(L, b);

      if (a==b) {
          done = true;
      }
      else {
          if ((done == false) && (a > b)) {
              myswap(a, b);
          }
          int old = atomicMin(&L[b], a+1);
          if (old == b + 1) {
              done = true;
          }
          b = old - 1;
      }
  }
}
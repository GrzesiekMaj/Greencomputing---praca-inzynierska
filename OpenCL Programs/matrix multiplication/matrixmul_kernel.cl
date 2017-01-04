
/* kernel.cl 
 * Matrix multiplication: C = A * B.
 * Device code.
 */
 
// OpenCL Kernel
__kernel void
matrixMul(__global float* C, 
          __global float* A, 
          __global float* B, 
          int wA, int wB, int card_num, int device_amount)
{
  
   int tx = get_global_id(0); // 0 1 2 3
   int ty = get_global_id(1); // 0 1
   float elementA;
   float elementB;
   // value stores the element that is 
   // computed by the thread
   float value = 0;
   for (int k = 0; k < (wA/device_amount); ++k)
   {
	  if(card_num == 0){
		  elementA = A[ty * wA + k];
		  elementB = B[k * wB + tx];
	  }
	  if(card_num == 1){
		  elementA = A[ty + ((wA/device_amount) + card_num) * wA + k];
		  elementB = B[k * wB + tx];
	  }
      value += elementA * elementB;
   }
 
   // Write the matrix to device memory each 
   // thread writes one element
   
   C[ty * wA + tx] = value;
}

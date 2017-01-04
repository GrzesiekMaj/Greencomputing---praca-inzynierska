////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

////////////////////////////////////////////////////////////////////////////////
#define WA 16384
#define HA 16384
#define WB 16384

#define HB WA
#define WC WB
#define HC HA
////////////////////////////////////////////////////////////////////////////////

 
// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = rand() / (float)RAND_MAX;
}

long LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}

int main(int argc, char** argv)
{
   int err;                            // error code returned from api call
   
    // OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;
   
   cl_mem d_A_clone;
   cl_mem d_B_clone;
   cl_mem d_C_clone;

   // set seed for rand()
   srand(2014);
 
   //Allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);

   //Initialize host memory
   randomMemInit(h_A, size_A);
   randomMemInit(h_B, size_B);
 
   //Allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);
   float* h_C_clone = (float*) malloc(mem_size_C);
  
   printf("Initializing OpenCL device...\n"); 

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
   
   cl_uint deviceNumber;
   err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNumber);
   printf("clGetDeviceIDs error code = %d\n", err);
   cl_device_id* device_id = new cl_device_id[deviceNumber];
   // Connect to a compute device
   
   err = clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, deviceNumber, device_id, &deviceNumber);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       return EXIT_FAILURE;
   }
   deviceNumber = 1;
   printf("Device amount:%d\n", deviceNumber);
   
   cl_context context[deviceNumber];
  
   // Create a compute context 
   for(int i = 0; i< deviceNumber; i++){
		context[i] = clCreateContext(0, 1, &device_id[i], NULL, NULL, &err);
   }
       printf("context error code = %d\n", err);

   // Create a command commands
   cl_command_queue commands[deviceNumber];  
   for(int i = 0; i< deviceNumber; i++){
		commands[i] = clCreateCommandQueue(context[i], device_id[i], 0, &err);
   }
   if (!commands[0])
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;
   cl_program program[deviceNumber];
   
   lFileSize = LoadOpenCLKernel("matrixmul_kernel.cl", &KernelSource);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }
	for(int i = 0; i< deviceNumber; i++){
		program[i] = clCreateProgramWithSource(context[i], 1, (const char **) & KernelSource, NULL, &err);
	}
   if (!program[0])
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }

   // Build the program executable
   for(int i = 0; i< deviceNumber; i++){
		err = clBuildProgram(program[i], 0, NULL, NULL, NULL, NULL);
   }
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program[0], device_id[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   cl_kernel kernel[deviceNumber];
   for(int i = 0; i< deviceNumber; i++){
		kernel[i] = clCreateKernel(program[i], "matrixMul", &err);
   }
   if (!kernel[0] || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context[0], CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context[0], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
   d_B = clCreateBuffer(context[0], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);
   
   d_C_clone = clCreateBuffer(context[1], CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A_clone = clCreateBuffer(context[1], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
   d_B_clone = clCreateBuffer(context[1], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);

   if (!d_A || !d_B || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", WA,HA,WB,HB); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wA = WA;
   int wC = WC;
   for(int i = 0; i< deviceNumber; i++){
	   if (i == 0){
		   err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&d_C);
		   err |= clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&d_A);
		   err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *)&d_B);
		   err |= clSetKernelArg(kernel[i], 3, sizeof(int), (void *)&wA);
		   err |= clSetKernelArg(kernel[i], 4, sizeof(int), (void *)&wC);
		   err |= clSetKernelArg(kernel[i], 5, sizeof(int), (void *)&i);
		   err |= clSetKernelArg(kernel[i], 6, sizeof(int), (void *)&deviceNumber);
	   }
	   if (i == 1){
		   err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&d_C_clone);
		   err |= clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&d_A_clone);
		   err |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *)&d_B_clone);
		   err |= clSetKernelArg(kernel[i], 3, sizeof(int), (void *)&wA);
		   err |= clSetKernelArg(kernel[i], 4, sizeof(int), (void *)&wC);
		   err |= clSetKernelArg(kernel[i], 5, sizeof(int), (void *)&i);
		   err |= clSetKernelArg(kernel[i], 6, sizeof(int), (void *)&deviceNumber);
	   }
   }

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }
 
   localWorkSize[0] = 1;
   localWorkSize[1] = 1;
   globalWorkSize[0] = WA;
   globalWorkSize[1] = WA/deviceNumber;
 
   for(int i = 0; i< deviceNumber; i++){
		err = clEnqueueNDRangeKernel(commands[i], kernel[i], 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   }

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }
 
   //Retrieve result from device
   for(int i = 0; i< deviceNumber; i++){
	   if(i == 0){
		err = clEnqueueReadBuffer(commands[i], d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
		}
	   if(i == 1){
		   err = clEnqueueReadBuffer(commands[i], d_C_clone, CL_TRUE, 0, mem_size_C, h_C_clone, 0, NULL, NULL);
	   }
   }

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }
 
   //print out the results

   printf("\n\nMatrix C (Results)\n");
  /*
   for(int i = 0; i < (size_C/deviceNumber); i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");
   if (deviceNumber == 2){
	   for(int i = 0; i < (size_C/deviceNumber); i++)
	   {
		  printf("%f ", h_C_clone[i]);
		  if(((i + 1) % WC) == 0)
		  printf("\n");
	   }
	   printf("\n");
   }
   */

  
   printf("Matrix multiplication completed...\n"); 

   //Shutdown and cleanup
   free(h_A);
   free(h_B);
   free(h_C);
   free(h_C_clone);
 
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);
   clReleaseMemObject(d_A_clone);
   clReleaseMemObject(d_C_clone);
   clReleaseMemObject(d_B_clone);

   for(int i = 0; i< deviceNumber; i++){
	   clReleaseProgram(program[i]);
	   clReleaseKernel(kernel[i]);
	   clReleaseCommandQueue(commands[i]);
	   clReleaseContext(context[i]);
   }

   return 0;
}

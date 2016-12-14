#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <ctime>
 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  const cl_int a,                       \n" \
"                       const cl_int b,                       \n" \
"						__global cl_float delta;						 \n" \
"                       __global cl_float *c,                       \n" \
"                       const cl_float dx)                       \n" \
"                                                                \n" \
"                                     							 \n" \
"    	int id = get_global_id(0);							     \n" \
"		c[i] = 0;						 						 \n" \
"                                                                \n "\
"  	 for(float i = id* delta; i < ((id+1) * delta-1); i+=dx){  	 \n" \
"	 	c[i] += (i + i+dx)*dx/2;								 \n" \
"																 \n" \
"																 \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;
 
int main( int argc, char* argv[] )
{
	
	// granica a  
	int boundry_a = atoi(argv[1]);

	// granica b
	int boundry_b = atoi(argv[2]);
	
	
	int boundry[2][2] = {{boundry_a, (boundry_b + boundry_a) / 2}, {(boundry_b + boundry_a) / 2, boundry_b}};
	
	// Length of vectors
	unsigned int size_of_problem = 1000000;
	printf("size of problem = %d\n", size_of_problem);
	cl_int err;
	cl_uint n_of_GPUs, num_platforms;
	// Bind to platform
	cl_platform_id cpPlatform;        // OpenCL platform
    err = clGetPlatformIDs(1, &cpPlatform, &num_platforms);
	printf("\nNumber of platforms:\t%u\n\n", num_platforms);

	// Get ID for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &n_of_GPUs);
	printf("n_of_GPUs = %d\n", n_of_GPUs);
	unsigned int n_of_kernels = 1024; // ilosc kerneli
	const int delta = (boundry_b - boundry_a)/n_of_GPUs/n_of_kernels;
    const float dx = delta/100000;
	
    // Host input vectors
    cl_uint *h_a[n_of_GPUs];
    cl_uint *h_b[n_of_GPUs];
    // Host output vector
    cl_uint *h_c[n_of_GPUs];
    // Device input buffers
    cl_mem d_a[n_of_GPUs];
    cl_mem d_b[n_of_GPUs];
    // Device output buffer
    cl_mem d_c[n_of_GPUs];
	
	cl_float* host_mem[2];
	
    
    cl_device_id device_id[n_of_GPUs];           // device ID
    cl_context context[n_of_GPUs];               // context
    cl_command_queue queue[n_of_GPUs];           // command queue
    cl_program program[n_of_GPUs];               // program
    cl_kernel kernel[n_of_GPUs];                 // kernel

	
    // Size, in bytes, of each vector
    //size_t bytes = n*sizeof(cl_uint);
	
	printf("bytes allocated per GPU = %d\n", bytes);
 
    // Allocate memory for each vector on host
	for(int i = 0; i<n_of_GPUs; i++){
		host_mem[i] = (cl_float*)malloc(sizeof(cl_float));
		//h_b[i] = (cl_uint*)malloc(bytes);
		//h_c[i] = (cl_uint*)malloc(bytes);
	}
			
		for(int i = 0; i < n; i++ )								
		{														
			/* h_a[0][i] = 1;											
			h_b[0][i] = 1;	
			h_a[1][i] = 1;											
			h_b[1][i] = 1;	 */
		}														
 
    size_t globalSize, localSize;

    //unsigned int n = 1000000/n_of_GPUs;
	
	// Number of work items in each local work group
    localSize = 1;
	
	printf("local work group on GPU: %d\n", localSize);
 
    // Number of total work items - localSize must be devisor
    globalSize = (boundry_b - boundry_a)/dx/1024;//ceil(n/(float)localSize)*localSize;
	clock_t start_cpu = clock();
	
	time_t start_wall = time(0);
	
	for(int i = 0; i<n_of_GPUs; i++){
		err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, n_of_GPUs, device_id, NULL);
	}

    // Create a context 
	for(int i = 0; i<n_of_GPUs; i++){
		context[i] = clCreateContext(0, 1, &device_id[i], NULL, NULL, &err);
	}
    // Create a command queue
	for(int i = 0; i<n_of_GPUs; i++){
		queue[i] = clCreateCommandQueue(context[i], device_id[i], 0, &err);
	}
 
    // Create the compute program from the source buffer
	for(int i = 0; i<n_of_GPUs; i++){
		program[i] = clCreateProgramWithSource(context[i], 1,
                            (const char **) & kernelSource, NULL, &err);
	}
 
    // Build the program executable 
	for(int i = 0; i<n_of_GPUs; i++){
		clBuildProgram(program[i], 0, NULL, NULL, NULL, NULL);
	}
 
    // Create the compute kernel in the program we wish to run
	for(int i = 0; i<n_of_GPUs; i++){
		kernel[i] = clCreateKernel(program[i], "vecAdd", &err);
	}

    // Create the input and output arrays in device memory for our calculation
	for(int i = 0; i<n_of_GPUs; i++){
		//d_a[i] = clCreateBuffer(context[i], CL_MEM_READ_ONLY , bytes, NULL, NULL);
		//d_b[i] = clCreateBuffer(context[i], CL_MEM_READ_ONLY , bytes, NULL, NULL);
		d_c[i] = clCreateBuffer(context[i], CL_MEM_WRITE_ONLY , 2*sizeof(float), NULL, NULL);
	}
 
    // Write our data set into the input array in device memory
	
	for(int i = 0; i<n_of_GPUs; i++){
/* 		err = clEnqueueWriteBuffer(queue[i], d_a[i],  CL_TRUE, 0,
									   bytes, h_a[i], 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(queue[i], d_b[i], CL_TRUE, 0,
									   bytes, h_b[i], 0, NULL, NULL); */
	}
	
	
    // Set the arguments to our compute kernel
	for(int i = 0; i<n_of_GPUs; i++){
		err  = clSetKernelArg(kernel[i], 0, sizeof(cl_uint), &boundry[i][0]);
		err |= clSetKernelArg(kernel[i], 1, sizeof(cl_uint), &boundry[i][1]);
		err |= clSetKernelArg(kernel[i], 2, sizeof(cl_float), &host_mem[i]);
		err |= clSetKernelArg(kernel[i], 3, sizeof(float), &delta);
		err |= clSetKernelArg(kernel[i], 4, sizeof(float), &dx);
	}
 
 

    // Execute the kernel over the entire range of the data set  
	for(int i = 0; i<n_of_GPUs; i++){
		err = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
	}
 
    // Wait for the command queue to get serviced before reading back results
	for(int i = 0; i<n_of_GPUs; i++){
		clFinish(queue[i]);
	}
 
    // Read the results from the device
	for(int i = 0; i<n_of_GPUs; i++){
    clEnqueueReadBuffer(queue[i], d_c[i], CL_TRUE, 0,
                                sizeof(float), &host_mem[i], 0, NULL, NULL );	

	}								
	
	print("Sum1: %f\n", host_mem[0]);
	print("Sum1: %f", host_mem[1]);
    //Sum up vector c and print result divided by n, this should equal 1 within error
    /* double sum  = 0;
	double sum2 = 0;

		for(int l = 0; l<n; l++){
			sum  += h_c[0][l];
			sum2 += h_c[1][l];
		}
		
    printf("final result 1st GPU: %d\n", sum);
	printf("final result 2nd GPU: %d\n", sum2);
    // release OpenCL resources */
	
	
	for(int i = 0; i<n_of_GPUs; i++){
    clReleaseMemObject(d_a[i]);
    clReleaseMemObject(d_b[i]);
    clReleaseMemObject(d_c[i]);
    clReleaseProgram(program[i]);
    clReleaseKernel(kernel[i]);
    clReleaseCommandQueue(queue[i]);
    clReleaseContext(context[i]);
	//release host memory
    free(h_a[i]);
    free(h_b[i]);
    free(h_c[i]);
	}
 
	clock_t end = clock();
	float seconds_cpu = (float)(end - start_cpu) / CLOCKS_PER_SEC;
	double seconds_wall = difftime( time(0), start_wall);	
	printf("time elapsed (CPU time): %f\n",seconds_cpu);
	printf("time elapsed (wall time): %f\n",seconds_wall);
 
    return 0;
}
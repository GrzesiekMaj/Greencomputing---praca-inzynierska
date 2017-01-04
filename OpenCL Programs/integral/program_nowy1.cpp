#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

int main()
{
	
	clock_t start_cpu = clock();
	
	time_t start_wall = time(0);
	
	
	cl_int error;
	cl_uint platformNumber;
	  // Kernel file descriptor 
	  FILE* fileKernel;
	  // kernel code string
	  char *kernelSource_str;
	  // Size of kernel source
		size_t source_size;
	
	error = clGetPlatformIDs(0, NULL, &platformNumber);
	printf("Number of platforms = %d\n", platformNumber);
	if (0 == platformNumber)
    {
        std::cout << "No OpenCL platforms found." << std::endl;

        return 0;
    }

	cl_platform_id platformId;
	
	error = clGetPlatformIDs(platformNumber, &platformId, NULL);
	
	cl_uint deviceNumber;

	cl_int boundary_a = 0;
	cl_int boundary_b = 134217728;
	
	
    error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNumber);

	// Get device identifiers.
    cl_device_id* deviceIds = new cl_device_id[deviceNumber];
	
    error = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, deviceNumber, deviceIds, &deviceNumber);
	std::cout<<"Device amount: "<<deviceNumber<<std::endl;
	deviceNumber = 2;
	// Allocate and initialize host arrays
    size_t global_size = 1;//134217728;//65536;
	std::cout<<"global_size: "<<global_size<<std::endl;
    size_t localWorkSize = 1;
	
    float* a = new float[1];
    float* b = new float[deviceNumber];
    float* c = new float[deviceNumber];
	*c = 0;
	
	*a = (boundary_b - boundary_a) / deviceNumber;        //delta - przedzial dla x GPU
	
	for(int i=0; i<deviceNumber; i++)
	{
	b[i] = i* (*a);
	std::cout<<"*b["<<i<<"] = "<<b[i]<<std::endl;
	}
	
	cl_context context[deviceNumber];
	
	for(int i = 0; i< deviceNumber; i++){
	context[i] = clCreateContext(0, 1, &deviceIds[i], NULL, NULL, NULL);
	}
	
	if (NULL == context)
    {
        std::cout << "Failed to create OpenCL context." << std::endl;
    }

    // Create a command-queue
	cl_command_queue commandQueue[deviceNumber];
	for(int i = 0; i< deviceNumber; i++){
		commandQueue[i] = clCreateCommandQueue(context[i], deviceIds[i], 0, &error);
	}
	
	
	
	std::cout<<"error: "<<error<<std::endl;
	// Allocate the OpenCL buffer memory objects for source and result on the device.
	cl_mem bufferA[deviceNumber];
	cl_mem bufferB[deviceNumber];
	cl_mem bufferC[deviceNumber];
	cl_mem bufferD[1024];
	for(int i = 0; i< deviceNumber; i++){
		bufferA[i] = clCreateBuffer(context[i], CL_MEM_READ_WRITE , sizeof(float) * global_size, NULL, &error);
		bufferB[i] = clCreateBuffer(context[i], CL_MEM_READ_ONLY , sizeof(float) * global_size, NULL, &error);
		bufferC[i] = clCreateBuffer(context[i], CL_MEM_WRITE_ONLY, sizeof(float) * global_size, NULL, &error);
		bufferD[i] = clCreateBuffer(context[i], CL_MEM_READ_WRITE, sizeof(float) * global_size, NULL, &error);
	}
	std::cout<<"clCreateBuffer: "<<error<<std::endl;
  // Load the kernel source code into the array kernelSource_str
  fileKernel = fopen("add1.cl", "r");
  if (!fileKernel) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }

  // Read kernel code
  kernelSource_str = (char*)malloc(MAX_SOURCE_SIZE*4);
  source_size = fread(kernelSource_str, 1, MAX_SOURCE_SIZE, fileKernel);
  fclose(fileKernel);
  
  cl_program program[deviceNumber];
  for(int i = 0; i< deviceNumber; i++){
	program[i] = clCreateProgramWithSource(context[i], 1, (const char **)&kernelSource_str, (const size_t *)&source_size, &error);
	error = clBuildProgram(program[i], 0, NULL, NULL, NULL, NULL);
  }
	std::cout<<"clBuildProgram: "<<error<<std::endl;
	char buffer[10240];
	clGetProgramBuildInfo(program[0], deviceIds[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
	fprintf(stderr, "CL Compilation failed:\n%s", buffer);

	// Create the kernel.
	cl_kernel kernel[deviceNumber];
	
	for(int i = 0; i< deviceNumber; i++){
		kernel[i] = clCreateKernel(program[i], "Add1", &error);
	}
	size_t workgroup_size_info;

		clGetKernelWorkGroupInfo(kernel[0],deviceIds[0],CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_info, NULL);
		std::cout<<"MAX_WORKGROUP_SIZE: "<<workgroup_size_info<<std::endl;
	
	
	
	std::cout<<"clCreateKernel: "<<error<<std::endl;
    // Set the Argument values.
	for(int i = 0; i< deviceNumber; i++){
		error = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void*)&bufferA[i]);
		error = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void*)&bufferB[i]);
		error = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void*)&bufferC[i]);
		error = clSetKernelArg(kernel[i], 3, sizeof(cl_int), (void*)&global_size);
		error = clSetKernelArg(kernel[i], 4, sizeof(cl_mem), (void*)&bufferD[i]);
	}
	std::cout<<"clSetKernelArg: "<<error<<std::endl;
	
    // Asynchronous write of data to GPU device.
	for(int i = 0; i< deviceNumber; i++){
		error = clEnqueueWriteBuffer(commandQueue[i], bufferA[i], CL_FALSE, 0, sizeof(float), a, 0, NULL, NULL);
		error = clEnqueueWriteBuffer(commandQueue[i], bufferB[i], CL_FALSE, 0, sizeof(float), &b[i], 0, NULL, NULL);
	}
	std::cout<<"clEnqueueWriteBuffer: "<<error<<std::endl;
	//error = clEnqueueWriteBuffer(commandQueue, bufferB, CL_FALSE, 0, sizeof(cl_int) * global_size, b[i], 0, NULL, NULL);
	
	// Launch kernel.
	for(int i = 0; i< deviceNumber; i++){
		error = clEnqueueNDRangeKernel(commandQueue[i], kernel[i], 1, NULL, &global_size, &localWorkSize, 0, NULL, NULL);
		std::cout<<"clEnqueueNDRangeKernel: "<<error<<std::endl;
	}
	//std::cout<<"clEnqueueNDRangeKernel: "<<error<<std::endl;
	
    // Read back results and check accumulated errors.
	for(int i = 0; i< deviceNumber; i++){
		std::cout<<"clEnqueueReadBuffer: "<<error<<std::endl;
    error = clEnqueueReadBuffer(commandQueue[i], bufferC[i], CL_TRUE, 0, sizeof(float), &c[i], 0, NULL, NULL);
	}
	//std::cout<<"clEnqueueReadBuffer: "<<error<<std::endl;

	float sum = 0;
    // Print results.
    for (size_t i = 0; i < deviceNumber; ++i)
    {
		std::cout<< "wynik = " << c[i] << std::endl;
    }
	for (size_t i = 0; i < deviceNumber; ++i)
    {
		sum += c[i];
    }
	
	std::cout<< "suma = " << sum << std::endl;
	
	/*for(int i = 0; i< deviceNumber; i++){
		clReleaseMemObject(bufferA[i]);
		clReleaseMemObject(bufferB[i]);
		clReleaseMemObject(bufferC[i]);
		clReleaseMemObject(bufferD[i]);
	}
	*/
	
	clock_t end = clock();
	float seconds_cpu = (float)(end - start_cpu) / CLOCKS_PER_SEC;
	double seconds_wall = difftime( time(0), start_wall);	
	printf("time elapsed (CPU time): %f\n",seconds_cpu);
	printf("time elapsed (wall time): %f\n",seconds_wall);
/*
	 //Cleanup and free memory.
	for(int i = 0; i< deviceNumber; i++){
		clReleaseKernel(kernel[i]);
		clReleaseProgram(program[i]);
		clReleaseMemObject(bufferA[i]);
		clReleaseMemObject(bufferB[i]);
		clReleaseMemObject(bufferC[i]);
		clReleaseCommandQueue(commandQueue[i]);
		clReleaseContext(context[i]);
	}
	
	free(kernelSource_str);
*/	
    return 0;
}
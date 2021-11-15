#include "CLPathFinder.h"


#define CHECK_OPENCL_STATUS(ret, string)   									                \
    do {                                                                                    \
		cl_int clRet = ret;                                                                 \
		if (clRet != CL_SUCCESS) {                                                          \
			std::cerr << string << " with error code" << "(" << ret << ")" << std::endl;    \
			return 1;                                                                       \
		}																	                \
	} while(false)                                                                           

int CLPathFinder::init(std::string path, cl_device_id device, int kernel_version)
{
	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create context");

	std::string kernel_source;
	
	if (get_kernel_source(path, kernel_source)) {
		return 1;
	}

	const size_t code_size = kernel_source.size();
	const char* code_string = kernel_source.data();

	program = clCreateProgramWithSource(context, 1, &code_string, &code_size, &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create program with given source");

	clStatus = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	CHECK_OPENCL_STATUS(clStatus, "Error during source compiling");

	iterKernel = clCreateKernel(program, "bellmanFordIter", &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create bellmanFordIter kernel");

	initKernel = clCreateKernel(program, "bellmanFordInit", &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create bellmanFordInit kernel");

	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create command queue");

	this->kernel_version = kernel_version;
	this->device = device;

	return 0;
}

int CLPathFinder::set_args(Graph& g, int start)
{
	edgesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int) * g.edges.size(), nullptr, &clStatus);	
	CHECK_OPENCL_STATUS(clStatus, "Cannot create edges buffer");
	
	clStatus = clEnqueueWriteBuffer(queue, edgesBuffer, CL_FALSE, 0, 2 * sizeof(int) * g.edges.size(), g.edges.data(), 0, nullptr, nullptr);
	CHECK_OPENCL_STATUS(clStatus, "Cannot write graph into edges buffer");

	weightsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(double) * g.weight.size(), nullptr, &clStatus);	
	CHECK_OPENCL_STATUS(clStatus, "Cannot create buffer for edges weights");
	
	clStatus = clEnqueueWriteBuffer(queue, weightsBuffer, CL_FALSE, 0, sizeof(double) * g.weight.size(), g.weight.data(), 0, nullptr, nullptr);
	CHECK_OPENCL_STATUS(clStatus, "Cannot write to weight buffer");

	distancesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * g.vertices_amount, nullptr, &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create distances buffer");

	distances.resize(g.vertices_amount);
	clStatus = clEnqueueWriteBuffer(queue, distancesBuffer, CL_FALSE, 0, sizeof(double) * distances.size(), distances.data(), 0, nullptr, nullptr);
	CHECK_OPENCL_STATUS(clStatus, "Cannot write into distances buffer");
	


	changed = 0;
	changedBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(changed), &changed, &clStatus);
	CHECK_OPENCL_STATUS(clStatus, "Cannot create changing check buffer");

	int arg = 0;

	clSetKernelArg(initKernel, arg++, sizeof(cl_mem), &distancesBuffer);
	clSetKernelArg(initKernel, arg++, sizeof(cl_uint), &g.vertices_amount);
	clSetKernelArg(initKernel, arg++, sizeof(cl_uint), &start);

	arg = 0;

	edges_amount = g.edges.size();

	clSetKernelArg(iterKernel, arg++, sizeof(cl_uint), &edges_amount);
	clSetKernelArg(iterKernel, arg++, sizeof(cl_mem), &edgesBuffer);
	clSetKernelArg(iterKernel, arg++, sizeof(cl_mem), &weightsBuffer);
	clSetKernelArg(iterKernel, arg++, sizeof(cl_mem), &distancesBuffer);
	clSetKernelArg(iterKernel, arg++, sizeof(cl_mem), &changedBuffer);

	size_t localSizes[3] = { 0 };
	clStatus = clGetKernelWorkGroupInfo(iterKernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(localSizes), localSizes, nullptr);
	
	CHECK_OPENCL_STATUS(clStatus, "Cannot get workgroup information from given kernel");
	
	localSize = localSizes[0];
	globalSizeInit = global_align(g.vertices_amount, localSize);
    globalSizeIter = global_align(g.edges.size(), localSize);

	if (kernel_version == 2) {
		globalSizeIter = global_align(globalSizeIter / localSize, localSize);
	}

	return 0;
}

int CLPathFinder::run()
{

	int count_iterations = 0;

	clEnqueueNDRangeKernel(queue, initKernel, 1, 0, &globalSizeInit, &localSize, 0, nullptr, nullptr);

	do {
		changed = 0;

		clEnqueueWriteBuffer(queue, changedBuffer, CL_FALSE, 0, sizeof(cl_uint), &changed, 0, nullptr, nullptr);

		clEnqueueNDRangeKernel(queue, iterKernel, 1, 0, &globalSizeIter, &localSize, 0, nullptr, nullptr);

		clEnqueueReadBuffer(queue, changedBuffer, CL_TRUE, 0, sizeof(cl_uint), &changed, 0, nullptr, nullptr);
		
		count_iterations++;
	} while (changed);

	clFinish(queue);
    clEnqueueReadBuffer(queue, distancesBuffer, CL_TRUE, 0,  sizeof(double) * distances.size(), distances.data(), 0, nullptr, nullptr);
    
	std::cout << "OpenCL FordBellman(ver" << kernel_version << ") works with " << count_iterations << " iterations." << std::endl;

	return 0;
}

void CLPathFinder::releaseResources()
{
	if (edgesBuffer) {
		clReleaseMemObject(edgesBuffer);
	}
    if (weightsBuffer) {
        clReleaseMemObject(weightsBuffer);
	}
    if (distancesBuffer) {
        clReleaseMemObject(distancesBuffer);
	}
    if (program) {
		clReleaseProgram(program);
	}
    if (initKernel) {
        clReleaseKernel(initKernel);
	}
    if (iterKernel) {
        clReleaseKernel(iterKernel);
	}
    if (queue) {
        clReleaseCommandQueue(queue);
	}
    if (context) {
        clReleaseContext(context);
	}
}

std::vector<double> CLPathFinder::get_distances()
{
	return distances;
}

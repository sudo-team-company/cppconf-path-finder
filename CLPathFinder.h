#pragma once
#include <string>
#include "graph.h"
#include <CL/cl.h>

class CLPathFinder 
{
public:
	CLPathFinder() :
		device(nullptr),
		context(nullptr),
		program(nullptr),
		iterKernel(nullptr),
		initKernel(nullptr),
		queue(nullptr),
		edgesBuffer(nullptr),
		weightsBuffer(nullptr),
		distancesBuffer(nullptr),
		changedBuffer(nullptr)
	{}

	int init(std::string path, cl_device_id device, int kernel_version);

	int set_args(Graph& g, int start);

	int run();

	void releaseResources();

	~CLPathFinder() {
		releaseResources();
	}

	std::vector<double> get_distances();

private:
	int global_align(int n, int size) {
		return n % size == 0 ? n : n - n % size + size;
	}

	int get_kernel_source(std::string path, std::string& code) {
		std::ifstream in;

		in.open(path);

		if (in.fail()) {
			std::cerr << "Can`t open cl source file for reading!" << std::endl;
			return 1;
		}

		code = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

		if (in.fail()) {
			std::cerr << "Can`t read cl source file!" << std::endl;
			return 1;
		}

		return 0;
	}

	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_kernel iterKernel;
	cl_kernel initKernel;
	cl_command_queue queue;

	cl_mem edgesBuffer;
	cl_mem weightsBuffer;
	cl_mem distancesBuffer;
	cl_mem changedBuffer;
	cl_uint changed;
	cl_uint edges_amount;

	std::vector<double> distances;

	cl_int clStatus;

	int kernel_version;
	size_t localSize;
	size_t globalSizeIter;
	size_t globalSizeInit;
};

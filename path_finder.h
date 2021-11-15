#pragma once
#include <CL/cl.h>
#include <iostream>
#include "graph.h"
#include <queue>
#include "CLPathFinder.h"

const double inf = 1e12;

typedef std::function<std::vector<double>(Graph&, int, std::map<std::string, std::vector<std::double_t>>&)> path_finder_algo;

std::vector<double> fordBellman(Graph& g, int start, std::map<std::string, std::vector<std::double_t>>& times) {
	std::cout << "Starting simple FordBellman algo" << std::endl;

	auto t_start = std::chrono::high_resolution_clock::now();

	std::vector<double> d(g.vertices_amount, inf);

	d[start] = 0.0;

	bool anyChange;

	int iteration_number = 0;

	do {
		anyChange = false;
		for (size_t i = 0; i < g.edges.size(); i++) {
			auto edge = g[i];

			if (d[edge.first] < inf + 1e-9) {
				if (d[edge.first] + g.weight[i] < d[edge.second]) {
					d[edge.second] = d[edge.first] + g.weight[i];
					anyChange = true;
				}
			}
		}

		iteration_number++;
	} while (anyChange);

	auto t_end = std::chrono::high_resolution_clock::now();

	std::cout << "FordBellman algo works with " << iteration_number << " iterations." << std::endl;

	const auto overall = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000;

	std::cout << "FordBellman time work is " << overall << std::endl;

	times["FordBellman(CPU)"].push_back(overall);

	return d;
}

std::vector<double> dijkstra(Graph& graph, int start, std::map<std::string, std::vector<std::double_t>>& times) {
	std::cout << "Starting Dijkstra's algo" << std::endl;

	auto t_start = std::chrono::high_resolution_clock::now();
	
	std::vector<double> d(graph.vertices_amount, inf);

	std::vector<std::vector<std::pair<double, int>>> g(graph.vertices_amount);

	for (int i = 0; i < graph.edges.size(); i++) {
		auto edge = graph.edges[i];
		g[edge.first].emplace_back(graph.weight[i], edge.second);
	}

	std::priority_queue<std::pair<double, int>> q;

	d[start] = 0.0;
	q.push(std::make_pair(0.0, start));

	while (!q.empty()) {
		auto v = q.top().second;
		auto weight = -q.top().first;

		q.pop();

		if (weight > d[v]) {
			continue;
		}

		for (size_t edge = 0; edge < g[v].size(); edge++) {
			auto to = g[v][edge].second;
			auto w = g[v][edge].first;
			
			if (d[v] + w < d[to]) {
				d[to] = d[v] + w;
				q.push(std::make_pair(-d[to], to));
			}
		}
	}

	auto t_end = std::chrono::high_resolution_clock::now();

	const auto overall = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000;

	times["Dijkstra(CPU)"].push_back(overall);

	std::cout << "Dijkstra time work is " << overall << std::endl;

	return d;
}

std::vector<cl_device_id> get_devices()
{
	cl_uint platforms_amount = 0;

	std::vector<cl_device_id> all_devices;

	cl_int clStatus = clGetPlatformIDs(0, nullptr, &platforms_amount);

	if (clStatus != CL_SUCCESS) {
		std::cerr << "No compatible OpenCL platforms found" << std::endl;
	}

	std::vector<cl_platform_id> platforms(platforms_amount);

	clGetPlatformIDs(platforms_amount, platforms.data(), nullptr);

	for (cl_uint i = 0; i < platforms_amount; i++) {
		cl_uint devices_per_platform;
		clStatus = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &devices_per_platform);

		std::vector<cl_device_id> devices(devices_per_platform);

		clStatus = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devices_per_platform, devices.data(), 0);

		all_devices.insert(all_devices.end(), devices.begin(), devices.end());
	}

	return all_devices;
}

bool get_device_name(cl_device_id device, std::string& arg_name, std::string& arg_vendor) {
	size_t string_size;
	cl_int clStatus;

	clStatus = clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &string_size);

	std::unique_ptr<char[]> vendor;

	try {
		vendor = std::unique_ptr<char[]>(new char[string_size]);
	}
	catch (...) {
		std::cerr << "Can`t allocate memory for string.";
	}

	clStatus = clGetDeviceInfo(device, CL_DEVICE_VENDOR, string_size, vendor.get(), nullptr);

	arg_vendor = std::string(vendor.get());

	clStatus = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &string_size);

	std::unique_ptr<char[]> name;

	try {
		name = std::unique_ptr<char[]>(new char[string_size]);
	}
	catch (...) {
		std::cerr << "Can`t allocate memory for string.";
	}

	clStatus = clGetDeviceInfo(device, CL_DEVICE_NAME, string_size, name.get(), nullptr);

	arg_name = std::string(name.get());
	return true;
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

std::vector<double> fordBellmanOpenCL(Graph& g, int start, std::map<std::string, std::vector<std::double_t>>& times) {
	std::cout << "Running " << __FUNCTION__ << std::endl << std::endl;

	auto devices = get_devices();

	std::vector<double> result;

	const size_t KERNELS_AMOUNT = 3;

	for (size_t kernel_version = 1; kernel_version <= KERNELS_AMOUNT; kernel_version++) {
		for (auto device : devices) {
			std::string device_vendor, device_name;
			if (!get_device_name(device, device_name, device_vendor)) {
				std::cerr << "Can`t get device name";
				continue;
			}

			if (device_name.find("Oclgrind") != std::string::npos || device_name.find("GT 710") != std::string::npos)
			{
				continue;
			}

			if (device_name.find("Intel") != std::string::npos) 
			{
				continue;           
			}

			std::cout << "Starting FordBellman(ver" << kernel_version <<") on " << device_name << "(" << device_vendor << ")" << std::endl;

			auto t_start = std::chrono::high_resolution_clock::now();

			CLPathFinder cl_path_finder;

			if (cl_path_finder.init("..\\fordBellman_ver" + std::to_string(kernel_version) + ".cl", device, kernel_version)) {
				continue;
			}

			if (cl_path_finder.set_args(g, start)) {
				continue;
			}

			auto t_kernel = std::chrono::high_resolution_clock::now();

			if (cl_path_finder.run()) {
				continue;
			}

			auto t_end = std::chrono::high_resolution_clock::now();

			const auto overall = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000;
			const auto kernels_time = std::chrono::duration<double, std::milli>(t_end - t_kernel).count() / 1000;

			std::cout << "Overall time: " << overall << "\t kernels time: " << kernels_time << std::endl;
			std::cout << std::endl;

			times[device_name + "(ver" + std::to_string(kernel_version) + ")"].push_back(overall);

			auto new_result = cl_path_finder.get_distances();

			if (!result.empty()) {
                for (int i = 0; i < result.size(); i++) {
                    if (std::abs(result[i] - new_result[i]) > 1e-6) {
						std::cerr << "OpenCl kernel result is invalid";
						return new_result;
					} 
				}
			}

			result = new_result;
		}
	}

	return result;
}

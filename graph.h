#pragma once
#include <vector>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <chrono>

class Graph 
{
public:

	int open(const char* file) {
		auto t_start = std::chrono::high_resolution_clock::now();

		std::mt19937 gen;

		std::ifstream fin(file);

		if (fin.fail()) {
			std::cerr << "Can`t open input file for reading graph!";
			return 1;
		}

		int m, from, to;

		fin >> vertices_amount >> m;

		std::unordered_set<int> count;

		for (int i = 0; i < m; i++) {
			fin >> from >> to;
			edges.emplace_back(from, to);
			count.insert(from);
			count.insert(to);
			double price = (double(gen()) / RAND_MAX) * 1e3;
			weight.push_back(price);
			edges.emplace_back(to, from);
			weight.push_back(weight.back());
		}

		int cnt = 0;
		std::unordered_map<int, int> rename;
		for (auto it = count.begin(); it != count.end(); it++) {
			rename[*it] = cnt;
			cnt++;
		}

		for (int i = 0; i < edges.size(); i++) {
			auto old = edges[i];
			old.first = rename[old.first];
			old.second = rename[old.second];
			edges[i] = old;
		}

		vertices_amount = count.size();

//#define SORT_FIRST
#ifdef SORT_FIRST
		std::vector<std::pair<std::pair<int, int>, double>> arr;
		for (int i = 0; i < edges.size(); i++)
		{
			arr.push_back({ { edges[i].first, edges[i].second }, weight[i] });
		}
		std::sort(arr.begin(), arr.end());
		for (int i = 0; i < arr.size(); i++)
		{
			edges[i] = arr[i].first;
			weight[i] = arr[i].second;
		}
#endif
//#define SORT_SECOND
#ifdef SORT_SECOND

		std::vector<std::pair<std::pair<int, int>, double>> arr;
		for (int i = 0; i < edges.size(); i++)
		{
			arr.push_back({ { edges[i].second, edges[i].first }, weight[i] });
		}
		std::sort(arr.begin(), arr.end());
		for (int i = 0; i < arr.size(); i++)
		{
			edges[i] = arr[i].first;
			std::swap(edges[i].first, edges[i].second);
			weight[i] = arr[i].second;
		}
#endif

		auto t_end = std::chrono::high_resolution_clock::now();

		const auto overall = std::chrono::duration<double, std::milli>(t_end - t_start).count() / 1000;

		std::cout << "Graph prepoccessing time: " << overall << std::endl << std::endl;

		return 0;
	}

	std::pair<int, int> operator[](const int i) const {
		return edges[i];
	}


	int vertices_amount; // Amount of vertices in graph.
	std::vector<std::pair<int, int>> edges; // Graph edges [begin, end].
	std::vector<double> weight; // Weight array of edges.
};

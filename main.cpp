#include <CL/cl.h>
#include <iostream>
#include <functional>
#include <numeric>
#include <chrono>
#include "graph.h"
#include <vector>
#include <map>
#include <iomanip>
#include <string>
#include "path_finder.h"

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "Should be only 2 arguments [file, num_test]";
        return 1;
    }

    Graph graph;
    
    if (graph.open(argv[1])) {
        return 1;
    }
   
    int tests_amount;
    try {
        tests_amount = std::stoi(argv[2]);
    }
    catch (std::exception& e) {
        std::cerr << "Second argument should be integer";
        return 1;
    }

    std::vector<path_finder_algo> algos = { fordBellman, dijkstra , fordBellmanOpenCL };
    std::map<std::string, std::vector<double>> working_times;
    std::vector<std::vector<double>> results;
    std::mt19937 gen;

    std::cout << "Running " << tests_amount << " tests on graph with " << graph.vertices_amount << " vertices and " << graph.edges.size() << " edges" << std::endl << std::endl;

    for (int test = 0; test < tests_amount; test++) {
        std::cout << "#Test num " << test + 1 << std::endl;

        int start = gen() % graph.vertices_amount;

        std::cout << "Begin vertex is " << start << std::endl << std::endl;

        for (auto& algo : algos) {
            std::vector<double> res;

            try {
                res = algo(graph, start, working_times);
            }
            catch (std::exception& e) {
                continue;
            }

            results.push_back(res);

            std::cout << std::endl;
        }

        for (int i = 1; i < algos.size(); i++) {
            for (int j = 0; j < results[0].size(); j++) {
                if (std::abs(results[0][j] - results[i][j]) > 1e-6) {
                    std::cerr << "Wrong algo work with index " << i << std::endl;
                    std::cout << "Distance to vertex " << j << " should be " << results[0][j] << " ,but was " << results[i][j] << std::endl;
                    return 1;
                }
            }
        }
    }

    std::map<std::string, double> average_times;
    

    for (auto& i : working_times) {
        double average_time = std::accumulate(i.second.begin(), i.second.end(), 0.0) / i.second.size();
        average_times[i.first] = average_time;
    }

    double max_time = average_times["Dijkstra(CPU)"];

    std::cout  <<"\nAverage times" << std::endl <<
        std::left << std::setw(40) << "Name" << std::setw(16) << "avg.time" << "percent" << std::endl;

    std::cout << std::fixed;

    for (auto& i : average_times) {
        std::cout << std::left << std::setw(33) << std::setprecision(2) << i.first << "\t" << std::setw(10) << i.second << "\t" << std::setprecision(0) << i.second / max_time * 100.0 << "%" << std::endl;
    }

}


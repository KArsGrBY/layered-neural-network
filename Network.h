#ifndef NN1_NETWORK_H
#define NN1_NETWORK_H

#include "memory"
#include "vector"
#include "neuron.h"
#include "iostream"
#include "fstream"
#include "iomanip"
#include "algorithm"

class Network {
public:
	Network (int inputLayerSize, std::vector <int> hideLayersSize, int outputLayerSize, std::shared_ptr <Activator> activator);

	Network (int inputLayerSize, std::vector <int> hideLayersSize, int outputLayerSize, std::shared_ptr <Activator> activator, double speed, double inertial);

	Network (std::string path);

	Network (std::string path, double speed, double inertial);

	void save (std::string path);

	void learning (const std::vector <std::pair <std::vector <double>, std::vector <double>>> & trainingSamples, double minimalError);

	std::vector <double> run (const std::vector <double> & input);

private:
	void setTrainingSample (const std::vector <double> & trainingSample);

	void iteration (const std::vector <double> & trainingSampleInput);

	void backPropagation (const std::vector <double> & trainingSampleOutput);

	double getSampleError (const std::vector <double> & trainingSampleOutput);

	std::vector <std::vector <std::shared_ptr <Neuron>>> neurons_;
	std::vector <std::shared_ptr <Neuron>> inputNeurons_, outputNeurons_;
	double speed_, inertial_;
	int synapsesCount_;
};


#endif //NN1_NETWORK_H
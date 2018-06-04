#ifndef NN1_NEURON_H
#define NN1_NEURON_H

#include "memory"
#include "vector"
#include "functional"
#include "activators.h"

class Neuron;

struct Synapse {
	std::shared_ptr <Neuron> neuron;
	double weight;

	Synapse (std::shared_ptr <Neuron> neuron) {
		this->weight = ((double) rand() / RAND_MAX) * 0.001;
		this->neuron = neuron;
	}

	Synapse (std::shared_ptr <Neuron> neuron, double weight) {
		this->weight = weight;
		this->neuron = neuron;
	}
};

class Neuron {
public:
	Neuron (std::shared_ptr <Activator> activator, int layer, int number);

	void setInputSignal (double inputSignal);

	double getInputSignal ();

	void pushSignal ();

	void save (std::ofstream & fout);

	void addSynapse (std::shared_ptr <Neuron> neuron);

	void addSynapse (std::shared_ptr <Neuron> neuron, double weight);

	void computeErrorSignal (const double *trainingSampleOutput, double inertial);

	void updateWeights (double speed);

	double getOutputSignal ();

	void computeOutputSignal ();

	int getLayer();

	int getNumber();

private:
	double inputSignal_, errorSignal_, outputSignal_;
	int layer_, number_;

	std::shared_ptr <Activator> activator_;

	std::vector <Synapse> synapses_;
};

#endif //NN1_NEURON_H

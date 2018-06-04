#include "Neuron.h"
#include "fstream"
#include "iostream"
#include "iomanip"
#include "Tools.h"

Neuron::Neuron (std::shared_ptr <Activator> activator, int layer, int number) {
	inputSignal_ = 0;
	errorSignal_ = 0;
	outputSignal_ = 0;
	activator_ = activator;
	layer_ = layer;
    number_ = number;
    synapses_.clear();
}

double Neuron::getInputSignal () {
	return inputSignal_;
}

int Neuron::getLayer () {
	return layer_;
}

int Neuron::getNumber () {
	return number_;
}

void Neuron::setInputSignal (double inputSignal) {
	inputSignal_ = inputSignal;
}

void Neuron::pushSignal () {
	for (const auto & synapse : synapses_) {
		synapse.neuron->setInputSignal(synapse.neuron->getInputSignal() + outputSignal_ * synapse.weight);
	}
}

void Neuron::addSynapse (std::shared_ptr <Neuron> neuron) {
	synapses_.push_back(Synapse(neuron));
}

void Neuron::addSynapse (std::shared_ptr <Neuron> neuron, double weight) {
    synapses_.push_back(Synapse(neuron, weight));
}

void Neuron::save (std::ofstream & fout) {
    for (size_t i = 0; i < synapses_.size(); i++) {
		write<int>(fout, layer_);
		write<int>(fout, number_);
		write<int>(fout, synapses_[i].neuron->getLayer());
		write<int>(fout, synapses_[i].neuron->getNumber());
		write<double>(fout, synapses_[i].weight);
    }
}

void Neuron::computeErrorSignal (const double *trainingSampleOutput, double inertial) {
	double d = activator_->derivative(inputSignal_);

	if (trainingSampleOutput == nullptr) {
		errorSignal_ = 0;
		for (Synapse synapse : synapses_) {
			errorSignal_ += synapse.neuron->errorSignal_ * synapse.weight;
		}
		errorSignal_ *= 2 * d * inertial;
	} else {
		errorSignal_ = 2 * d * (outputSignal_ - * trainingSampleOutput) * inertial;
	}
}

void Neuron::updateWeights (double speed) {
	for (Synapse & synapse : synapses_) {
		synapse.weight += -synapse.neuron->errorSignal_ * outputSignal_ * speed;
	}
}

double Neuron::getOutputSignal () {
	return outputSignal_;
}

void Neuron::computeOutputSignal () {
	outputSignal_ = activator_->function(inputSignal_);
}

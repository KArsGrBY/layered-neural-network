#include "Network.h"
#include "Tools.h"

Network::Network (int inputLayerSize, std::vector <int> hideLayersSize, int outputLayerSize,
				  std::shared_ptr <Activator> activator) {
	neurons_.resize(hideLayersSize.size() + 2);
	inertial_ = 0.1;
	speed_ = 0.1;
	synapsesCount_ = 0;

	for (size_t i = 0; i < inputLayerSize; i++) { // add input neurons
		inputNeurons_.push_back(std::make_shared <Neuron>(Activators::linear, 0, i));
		neurons_[0].push_back(inputNeurons_.back());
	}

	for (size_t i = 0; i < outputLayerSize; i++) { // add output neurons
		outputNeurons_.push_back(std::make_shared <Neuron>(activator, hideLayersSize.size() + 1, i));
		neurons_.back().push_back(outputNeurons_.back());
	}
	for (size_t layer = 0; layer < hideLayersSize.size(); layer++) { // add hide layer's neurons
		for (size_t i = 0; i < hideLayersSize[layer]; i++) {
			neurons_[layer + 1].push_back(std::make_shared <Neuron>(activator, layer + 1, i));
		}
	}

	for (size_t layer = 0; layer + 1 < neurons_.size(); layer++) { // add synapses
		for (size_t i = 0; i < neurons_[layer].size(); i++) {
			for (size_t j = 0; j < neurons_[layer + 1].size(); j++) {
				neurons_[layer][i]->addSynapse(neurons_[layer + 1][j]);
				synapsesCount_++;
			}
		}

		std::shared_ptr <Neuron> neuron = std::make_shared <Neuron>(Activators::unit, layer, neurons_[layer].size()); // add bias neurons with synapses
		for (size_t i = 0; i < neurons_[layer + 1].size(); i++) {
			neuron->addSynapse(neurons_[layer + 1][i]);
			synapsesCount_++;
		}
		neurons_[layer].push_back(neuron);
	}
}

Network::Network (int inputLayerSize, std::vector <int> hideLayersSize, int outputLayerSize,
				  std::shared_ptr <Activator> activator, double speed, double inertial)
		: Network(inputLayerSize, hideLayersSize, outputLayerSize, activator) {
	speed_ = speed;
	inertial_ = inertial;
}


Network::Network (std::string path) {
    int inputLayerSize, outputLayerSize;
	std::vector <int> hideLayersSize;
	std::shared_ptr <Activator> activator = Activators::tanh;

    std::ifstream fin(path, std::ios::binary);

	inputLayerSize = read<int>(fin);
    outputLayerSize = read<int>(fin);
	hideLayersSize = std::vector<int>(read<int>(fin));

	for (size_t i = 0; i < hideLayersSize.size(); i++) {
        hideLayersSize[i] = read<int>(fin);
	}
	speed_ = read<double>(fin);
	inertial_ = read<double>(fin);
	synapsesCount_ = read<int>(fin);


	//fill
    neurons_.resize(hideLayersSize.size() + 2);
	for (size_t i = 0; i < inputLayerSize; i++) { // add input neurons
		inputNeurons_.push_back(std::make_shared <Neuron>(Activators::linear, 0, i));
		neurons_[0].push_back(inputNeurons_.back());
    }

    for (size_t i = 0; i < outputLayerSize; i++) { // add output neurons
		outputNeurons_.push_back(std::make_shared <Neuron>(activator, hideLayersSize.size() + 1, i));
		neurons_.back().push_back(outputNeurons_.back());
    }
	for (size_t layer = 0; layer < hideLayersSize.size(); layer++) { // add hide layer's neurons
		for (size_t i = 0; i < hideLayersSize[layer]; i++) {
			neurons_[layer + 1].push_back(std::make_shared <Neuron>(activator, layer + 1, i));
		}
	}

	for (size_t layer = 0; layer + 1 < neurons_.size(); layer++) { // add synapses
		std::shared_ptr <Neuron> neuron = std::make_shared <Neuron>(Activators::unit, layer, neurons_[layer].size()); // add bias neurons with synapses
		neurons_[layer].push_back(neuron);
	}

    for (size_t i = 0; i < synapsesCount_; i++) {
		int fromLayer, fromNumber, toLayer, toNumber;
        double weight;
		fromLayer = read<int>(fin);
		fromNumber = read<int>(fin);
		toLayer = read<int>(fin);
		toNumber = read<int>(fin);
        weight = read<double>(fin);
        neurons_[fromLayer][fromNumber]->addSynapse(neurons_[toLayer][toNumber], weight);
    }


	fin.close();
}

Network::Network (std::string path, double speed, double inertial) : Network(path) {
	speed_ = speed;
	inertial_ = inertial_;
}

void Network::save (std::string path) {
    std::ofstream fout(path, std::ios::binary);

    write<int>(fout, inputNeurons_.size());
	write<int>(fout, outputNeurons_.size());
	write<int>(fout, neurons_.size() - 2);

	for (size_t i = 1; i + 1 < neurons_.size(); i++) {
        write<int>(fout, neurons_[i].size() - 1);
	}
	write<double>(fout, speed_);
	write<double>(fout, inertial_);
	write<int>(fout, synapsesCount_);

	//fill

	for (size_t layer = 0; layer + 1 < neurons_.size(); layer++) { // add synapses
        for (size_t i = 0; i < neurons_[layer].size(); i++) {
            neurons_[layer][i]->save(fout);
		}
	}

	fout.close();
}

void Network::setTrainingSample (const std::vector <double> & trainingSample) {
	for (size_t i = 0; i < inputNeurons_.size(); i++) {
		inputNeurons_[i]->setInputSignal(trainingSample[i]);
	}
}

void Network::iteration (const std::vector <double> & trainingSampleInput) {
	setTrainingSample(trainingSampleInput);
	for (size_t layer = 0; layer + 1 < neurons_.size(); layer++) {
		for (size_t i = 0; i < neurons_[layer + 1].size(); i++) {
			neurons_[layer + 1][i]->setInputSignal(0);
		}
		for (size_t i = 0; i < neurons_[layer].size(); i++) {
			neurons_[layer][i]->computeOutputSignal();
			neurons_[layer][i]->pushSignal();
		}
	}
	for (size_t i = 0; i < outputNeurons_.size(); i++) {
		outputNeurons_[i]->computeOutputSignal();
	}
}

void Network::learning (const std::vector <std::pair <std::vector <double>, std::vector <double>>> & trainingSamples,
						double minimalError) {
	size_t epoch = 0;
	double error;
	std::vector <int> perm;
	for (size_t i = 0; i < trainingSamples.size(); i++) {
		perm.push_back(i);
	}

	do {
		std::random_shuffle(perm.begin(), perm.end());
		for (size_t sample = 0; sample < trainingSamples.size(); sample++) {
			iteration(trainingSamples[perm[sample]].first);
			backPropagation(trainingSamples[perm[sample]].second);
		}

		error = 0;
		for (size_t sample = 0; sample < trainingSamples.size(); sample++) {
			iteration(trainingSamples[sample].first);
			error += getSampleError(trainingSamples[sample].second);
        }

		if (epoch % 5 == 0) {
			std::cout << "epoch " << epoch << ", error is " << std::fixed << std::setprecision(15) << error << '\n' << '\r';
		}

		epoch++;

	} while (error > minimalError);
}

void Network::backPropagation (const std::vector <double> & trainingSampleOutput) {
	for (size_t i = 0; i < outputNeurons_.size(); i++) {
		outputNeurons_[i]->computeErrorSignal(& trainingSampleOutput[i], inertial_);
	}
	for (size_t layer = neurons_.size() - 1; layer > 0; layer--) {
		for (size_t i = 0; i < neurons_[layer - 1].size(); i++) {
			neurons_[layer - 1][i]->computeErrorSignal(nullptr, inertial_);
			neurons_[layer - 1][i]->updateWeights(speed_);
		}
	}
}

double Network::getSampleError (const std::vector <double> & trainingSampleOutput) {
	double sampleError = 0;
	for (size_t i = 0; i < outputNeurons_.size(); i++) {
		sampleError += pow(outputNeurons_[i]->getOutputSignal() - trainingSampleOutput[i], 2);
	}
	return sqrt(sampleError);
}

std::vector <double> Network::run (const std::vector <double> & input) {
	iteration(input);
	std::vector <double> output;
	for (const auto & outputNeuron : outputNeurons_) {
		output.push_back(outputNeuron->getOutputSignal());
	}
	return output;
}

#pragma once
#include <iostream>
#include <vector>
#include "Layer.h"
#include "Neuron.h"
class OutputLayer : public Layer
{
public:
	OutputLayer();
	~OutputLayer();

	OutputLayer& initLayer(OutputLayer& outputLayer);
	void printLayer(OutputLayer& outputLayer);

private:

};

inline OutputLayer& OutputLayer::initLayer(OutputLayer& outputLayer) {
	std::vector<double> listOfWeightOutTemp;
	std::vector<Neuron> listOfNeurons;

	for (size_t i = 0; i < outputLayer.numberOfNeuronInLayer; i++) {
		Neuron neuron;
		listOfWeightOutTemp.push_back(neuron.initNeuron());

		neuron.listOfWeightOut = listOfWeightOutTemp;
		listOfNeurons.push_back(neuron);

		listOfWeightOutTemp.clear();
	}

	outputLayer.listOfNeurons = listOfNeurons;
	return outputLayer;
}

inline void OutputLayer::printLayer(OutputLayer& outputLayer) {
	std::cout << "### Output Layer ###" << std::endl;
	int n = 1;
	for (Neuron& neuron : outputLayer.listOfNeurons) {
		std::cout << "Neuron #" << n << ":" << std::endl;
		std::cout << "Input Weights:" << std::endl;
		std::vector<double> weights = neuron.listOfWeightOut;
		for (double weight : weights) {
			std::cout << weight << std::endl;

		}
		n++;
	}
}

OutputLayer::OutputLayer()
{
}

OutputLayer::~OutputLayer()
{
}

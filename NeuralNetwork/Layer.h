#pragma once
#include <vector>
#include "Neuron.h"

class Layer
{
public:
	Layer();
	~Layer();

	void printLayer() const {}
	std::vector<Neuron> listOfNeurons;
	size_t numberOfNeuronInLayer;


private:
	

};

Layer::Layer()
{
}

Layer::~Layer()
{
}

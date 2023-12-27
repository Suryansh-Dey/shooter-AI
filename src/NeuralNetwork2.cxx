#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
/*
	My data structure for neural network
	[
	[number of layers in the neural network],[
		[number of neurons in layer1],[ [length of neuron1],[activation],[weight1],[weight2]....[bias]],			[[length of neuron2],[activation],[weight1],[weight2]....[bias] ]
.....],
	[number of layers in the neural network],[
		[number of neurons in layer2],[ [length of neuron1],[activation],[weight1],[weight2]....[bias]],			[[length of neuron2],[activation],[weight1],[weight2]....[bias] ]
.....],
.
.
.
	]
*/
typedef double ***NN;
enum NN_activation_types
{
	NN_type_Linear,
	NN_type_ReLu,
	NN_type_Sigmoid,
	NN_type_Softmax
};
enum NN_cost_types
{
	NN_type_meanSquareError,
	NN_type_crossEntropy
};
void NN_clear(double ***nn, double default_value = 0)
{
	for (int layer_no = 1; layer_no < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
			{
				*(*(*(nn + layer_no) + neuron_no) + weight_no) = default_value;
			}
		}
	}
}
double ***NN_create(const int *neuron_cts)
{
	int prev_neuron_no = -1;
	double ***nn = (double ***)(malloc((neuron_cts[0]) * 8));
	if (nn == NULL)
	{
		printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
		exit(1);
		return nn;
	}
	nn[0] = (double **)(malloc(8));
	nn[0][0] = (double *)(malloc(8));
	nn[0][0][0] = neuron_cts[0];
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		nn[layer_no] = (double **)(malloc((neuron_cts[layer_no] + 1) * 8));
		if (nn[layer_no] == NULL)
		{
			printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
			return nn;
			exit(1);
		}
		nn[layer_no][0] = (double *)(malloc(8 * 2));
		nn[layer_no][0][0] = neuron_cts[layer_no] + 1;
		nn[layer_no][0][1] = NN_type_ReLu;
		for (int neuron_no = 1; neuron_no <= neuron_cts[layer_no]; neuron_no++)
		{
			nn[layer_no][neuron_no] = (double *)(malloc((prev_neuron_no + 3) * 8));
			if (nn[layer_no][neuron_no] == NULL)
			{
				printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
				exit(1);
				return nn;
			}
			nn[layer_no][neuron_no][0] = prev_neuron_no + 2 + 1;
		}
		prev_neuron_no = neuron_cts[layer_no];
	}
	nn[int(nn[0][0][0]) - 1][0][1] = NN_type_Sigmoid;
	return nn;
}
double ***NN_create(std::vector<int> neuron_cts)
{
	int prev_neuron_no = -1;
	double ***nn = (double ***)(malloc((neuron_cts.size() + 1) * 8));
	if (nn == NULL)
	{
		printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
		exit(1);
		return nn;
	}
	nn[0] = (double **)(malloc(8));
	nn[0][0] = (double *)(malloc(8));
	nn[0][0][0] = neuron_cts.size() + 1;
	for (int layer_no = 1; layer_no < neuron_cts.size() + 1; layer_no++)
	{
		nn[layer_no] = (double **)(malloc((neuron_cts[layer_no - 1] + 1) * 8));
		if (nn[layer_no] == NULL)
		{
			printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
			return nn;
			exit(1);
		}
		nn[layer_no][0] = (double *)(malloc(8 * 2));
		nn[layer_no][0][0] = neuron_cts[layer_no - 1] + 1;
		nn[layer_no][0][1] = NN_type_ReLu;
		for (int neuron_no = 1; neuron_no <= neuron_cts[layer_no - 1]; neuron_no++)
		{
			nn[layer_no][neuron_no] = (double *)(malloc((prev_neuron_no + 3) * 8));
			if (nn[layer_no][neuron_no] == NULL)
			{
				printf("\nERROR: Memory was not allocated by OS to create the neural network\nNN_create() Aborted\n");
				exit(1);
				return nn;
			}
			nn[layer_no][neuron_no][0] = prev_neuron_no + 2 + 1;
		}
		prev_neuron_no = neuron_cts[layer_no - 1];
	}
	nn[int(nn[0][0][0]) - 1][0][1] = NN_type_Sigmoid;
	return nn;
}
inline int NN_layerCount(double ***nn)
{
	return int(nn[0][0][0]) - 1;
}
inline int NN_neuronCount(double ***nn, int layer_no)
{
	if ((layer_no > 0) and (layer_no < nn[0][0][0]))
		return int(nn[layer_no][0][0]) - 1;
	else
	{
		printf("\nERROR: The neural network had %i layers but number of neurons of %ith layer was asked\nNN_neuronCount() Aborted\n", NN_layerCount(nn), layer_no);
		exit(1);
		return -1;
	}
}
void NN_show(double ***nn)
{
	char Linear[] = "Linear", ReLu[] = "ReLu", Sigmoid[] = "Sigmoid", Softmax[] = "Softmax";
	char *activation_strs[] = {Linear, ReLu, Sigmoid, Softmax};
	printf("Number of layers in this neural network = %i\n", (int)(nn[0][0][0] - 1));
	printf("\n<<Layer 1 (input layer) has %i neurons>>\n\n", (int)(nn[1][0][0] - 1));
	for (int neuron_no = 1; neuron_no < nn[1][0][0]; neuron_no++)
	{
		printf("Neuron %i Activation- %f\n", neuron_no, nn[1][neuron_no][1]);
	}
	for (int layer_no = 2; layer_no < nn[0][0][0]; layer_no++)
	{
		if (layer_no != nn[0][0][0] - 1)
			printf("\n<<Layer %i has %i %s neurons>>\n\n", layer_no, (int)(nn[layer_no][0][0] - 1), activation_strs[int(nn[layer_no][0][1])]);
		else
			printf("\n<<Layer %i (output layer) has %i %s neurons>>\n\n", layer_no, (int)(nn[layer_no][0][0] - 1), activation_strs[int(nn[layer_no][0][1])]);
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			printf("Neuron %i\nActivation- %f\nWeights- [", neuron_no, nn[layer_no][neuron_no][1]);
			int weight_no = 2;
			for (; weight_no < nn[layer_no][neuron_no][0] - 1; weight_no++)
			{
				printf(" %f,", nn[layer_no][neuron_no][weight_no]);
			}
			printf("]\nBias- %f\n\n", nn[layer_no][neuron_no][weight_no]);
		}
	}
	printf("Network finished!\n");
}
inline double ***NN_dublicate(double ***nn)
{
	int neuron_cts[int(nn[0][0][0])];
	neuron_cts[0] = int(nn[0][0][0]);
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		neuron_cts[layer_no] = NN_neuronCount(nn, layer_no);
	}
	return NN_create(neuron_cts);
}
void NN_copy(double ***nn2, double ***nn)
{
	for (int layer_no = 1; (layer_no < ***nn) && (layer_no < ***nn2); layer_no++)
	{
		for (int neuron_no = 1; (neuron_no < ***(nn + layer_no)) && (neuron_no < ***(nn2 + layer_no)); neuron_no++)
		{
			for (int weight_no = 1; (weight_no < **(*(nn + layer_no) + neuron_no)) && (weight_no < **(*(nn2 + layer_no) + neuron_no)); weight_no++)
			{
				*(*(*(nn2 + layer_no) + neuron_no) + weight_no) = *(*(*(nn + layer_no) + neuron_no) + weight_no);
			}
		}
		nn2[layer_no][0][1] = nn[layer_no][0][1];
	}
}
void NN_delete(double ***nn)
{
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			free(nn[layer_no][neuron_no]); // = (double *)(malloc((prev_neuron_no + 3) * 8));
		}
		free(nn[layer_no][0]); // = (double *)(malloc(8));
		free(nn[layer_no]);	   // = (double **)(malloc((neuron_cts[layer_no] + 1) * 8));
	}
	free(nn[0][0]); // = (double *)(malloc(8));
	free(nn[0]);	// = (double **)(malloc(8));
	free(nn);		// = (double ***)(malloc((neuron_cts[0] + 1) * 8));
	nn = NULL;
}
inline void NN_resize(double ***nn, int *neuron_cts)
{
	double ***temp_nn = NN_create(neuron_cts);
	NN_copy(temp_nn, nn);
	NN_delete(nn);
	nn = NN_create(neuron_cts);
	NN_copy(nn, temp_nn);
}
inline void NN_input(double ***nn, double *input)
{
	for (int neuron_no = 1; neuron_no < ***(nn + 1); neuron_no++)
		*(*(*(nn + 1) + neuron_no) + 1) = *(input + neuron_no);
}
inline void NN_input(double ***nn, std::vector<double> input)
{
	for (int neuron_no = 1; neuron_no < nn[1][0][0]; neuron_no++)
		nn[1][neuron_no][1] = input[neuron_no - 1];
}
double NN_LEAK = 0.01, NN_LEARNING_RATE = 0.01, NN_EXPLODING_THRESHOLD = 1;
inline void NN_set_learningRate(double learning_rate)
{
	if (learning_rate <= 0)
	{
		printf("\nERROR: Non-positive value was passed to NN_set_learningRate\nNN_set_learningRate() Aborted\n");
		exit(1);
	}
	NN_LEARNING_RATE = learning_rate;
}
inline void NN_set_leak(double leak)
{
	NN_LEAK = leak;
}
inline double NN_Linear(double activation)
{
	return activation;
}
inline double NN_ReLu(double activation)
{
	return (activation < 0) * activation * NN_LEAK + (activation >= 0) * activation;
}
inline double NN_Sigmoid(double activation)
{
	return 1 / (pow(2.7, -activation) + 1);
}
inline double NN_Softmax(double activation)
{
	return activation;
}
double (*NN_activation_funcs[])(double){NN_Linear, NN_ReLu, NN_Sigmoid, NN_Softmax};
inline double NN_activationf(double activation, double **layer)
{
	return NN_activation_funcs[int(layer[0][1])](activation);
}
inline double NN_a_LinearDerivative(double activation)
{
	return 1;
}
inline double NN_a_ReLuDerivative(double activation)
{
	return (activation >= 0) + (activation < 0) * NN_LEAK;
}
inline double NN_a_SigmoidDerivative(double activation)
{
	return activation - activation * activation;
}
inline double NN_a_SoftmaxDerivative(double activation)
{
	return 1; // we have Handeled Softmax derivative * derivative of cost function (crossEntropy) in derivative of costFuncton to skip computation by storing derivative of cost w.r.t input of neuron instead of w.r.t activation of neuron at output layer's neurones of gradient data structure.
}
double (*NN_activationDerivative_funcs[])(double){NN_a_LinearDerivative, NN_a_ReLuDerivative, NN_a_SigmoidDerivative, NN_a_SoftmaxDerivative};
inline double NN_aActivationDerivativef(double activation, double **layer)
{
	return NN_activationDerivative_funcs[int(layer[0][1])](activation);
}

void NN_SoftmaxManager(double **layer)
{
	double max_activation = layer[1][1];
	for (int neuron_no = 2; neuron_no < **(layer); neuron_no++)
	{
		if (layer[neuron_no][1] > max_activation)
			max_activation = layer[neuron_no][1];
	}
	for (int neuron_no = 1; neuron_no < **(layer); neuron_no++)
		layer[neuron_no][1] = pow(2.7, layer[neuron_no][1] - max_activation);
	double sigma = 0;
	for (int neuron_no = 1; neuron_no < **(layer); neuron_no++)
		sigma += layer[neuron_no][1];
	for (int neuron_no = 1; neuron_no < **(layer); neuron_no++)
		layer[neuron_no][1] /= sigma;
}
inline void NN_cmpt_activation(double **prev_layer, double **current_layer, double *neuron)
{
	double activation = 0;
	int neuron_no = 1;
	for (; neuron_no < **prev_layer; neuron_no++)
	{
		activation += *(*(prev_layer + neuron_no) + 1) * (*(neuron + neuron_no + 1)); //Adding weigted sum
	}
	activation += *(neuron + neuron_no + 1); //Adding bias
	*(neuron + 1) = NN_activationf(activation, current_layer);
}
double **NN_process(double ***nn)
{
	int layer_no = 2;
	for (; layer_no < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			NN_cmpt_activation(*(nn + layer_no - 1), *(nn + layer_no), *(*(nn + layer_no) + neuron_no));
		}
	}
	if (nn[layer_no - 1][0][1] == NN_type_Softmax)
		NN_SoftmaxManager(nn[layer_no - 1]);
	return *(nn + (int)(***nn) - 1);
}
void NN_set_costFunction(double ***nn, enum NN_cost_types cost_type)
{
	if (nn[NN_layerCount(nn)][0][1] == NN_type_Softmax && cost_type != NN_type_crossEntropy)
	{
		printf("\nWARNING: NeuralNetwork2 only provides NN_type_crossEntropy with Softmax function in output layer\nNN_set_costFunction() Failed\n");
		return;
	}
	nn[1][0][1] = cost_type;
}
void NN_set_activationFunction(double ***nn, int layer_no, enum NN_activation_types activation_type)
{
	if (layer_no > NN_layerCount(nn))
	{
		printf("\nWARNING: Cannot set activation function of %ith layer. NOTE the passed nn has %i layers only\nNN_set_activationFunction() Failed\n", layer_no, NN_layerCount(nn));
		return;
	}
	if (layer_no <= 1)
	{
		printf("\nWARNING: Cannot set activation function of %ith layer. NOTE 1st layer ( input layer ) doesn't have activation function\nNN_set_activationFunction() Failed\n", layer_no);
		return;
	}
	if (activation_type == NN_type_Softmax)
	{
		if (layer_no == NN_layerCount(nn))
			NN_set_costFunction(nn, NN_type_crossEntropy);
		else
		{
			printf("\nWARNING: NeuralNetwork2 allows Softmax function ONLY on output layer.NOTE the passed nn has %i layers only\nNN_set_activationFunction() Failed\n", NN_layerCount(nn));
			return;
		}
	}
	nn[layer_no][0][1] = activation_type;
}
std::vector<double> NN_vectorOutput(double **layer)
{
	std::vector<double> output_vector(layer[0][0] - 1);
	for (int neuron_no = 1; neuron_no < layer[0][0]; neuron_no++)
		output_vector[neuron_no - 1] = layer[neuron_no][1];
	return output_vector;
}
void NN_save(double ***nn, const char *path)
{
	std::ofstream saving_file(path);
	saving_file << nn[0][0][0] << '\n';
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		saving_file << nn[layer_no][0][0] << '\n';
	}
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		saving_file << nn[layer_no][0][1] << ' ';
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 1; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				saving_file << nn[layer_no][neuron_no][weight_no] << ' ';
			}
		}
	}
}
double ***NN_load(const char *path)
{
	std::string word;
	std::ifstream loading_file(path);
	loading_file >> word;
	int neuron_cts[stoi(word)];
	neuron_cts[0] = stoi(word);
	for (int layer_no = 1; layer_no < neuron_cts[0]; layer_no++)
	{
		loading_file >> word;
		neuron_cts[layer_no] = stoi(word) - 1;
	}
	double ***nn = NN_create(neuron_cts);
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		loading_file >> word;
		nn[layer_no][0][1] = stod(word);
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 1; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				loading_file >> word;
				nn[layer_no][neuron_no][weight_no] = stod(word);
			}
		}
	}
	return nn;
}
double NN_MAX_WEIGHT = 1;
void NN_set_maxWeight(double maxWeight)
{
	if (maxWeight < 0)
	{
		printf("\nERROR: Negative value was passed to NN_set_maxWeight\nNN_set_maxWeight() Aborted\n");
		exit(1);
	}
	NN_MAX_WEIGHT = maxWeight;
}
void NN_mutate(double ***nn, double probability, double max_dweight)
{
	for (int layer_no = 2; layer_no < ***nn; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
			{
				double random_dweight = max_dweight * ((2 * (double)(rand()) / INT_MAX) - 1);
				if (((double)(rand()) / INT_MAX) < probability)
					*(*(*(nn + layer_no) + neuron_no) + weight_no) += random_dweight;
				if (abs(*(*(*(nn + layer_no) + neuron_no) + weight_no)) > NN_MAX_WEIGHT)
					*(*(*(nn + layer_no) + neuron_no) + weight_no) -= random_dweight;
			}
		}
	}
}
void NN_mutate_layer(double ***nn, double probability, double max_dweight, int layer_no)
{
	if (layer_no > NN_layerCount(nn))
	{
		printf("\nWARNING: Cannot mutate %ith layer. NOTE the passed nn has %i layers only\nNN_mutate_layer() Failed\n", layer_no, NN_layerCount(nn));
		return;
	}
	if (layer_no <= 1)
	{
		printf("\nWARNING: Cannot mutate %ith layer. NOTE 1st layer ( input layer ) doesn't contain weights but output layer do\nNN_mutate_layer() Failed\n", layer_no);
		return;
	}
	for (int neuron_no = 1; neuron_no < ***(nn + layer_no); neuron_no++)
	{
		for (int weight_no = 2; weight_no < *(*(*(nn + layer_no) + neuron_no)); weight_no++)
		{
			double random_dweight = max_dweight * ((2 * (double)(rand()) / INT_MAX) - 1);
			if (((double)(rand()) / INT_MAX) < probability)
				*(*(*(nn + layer_no) + neuron_no) + weight_no) += random_dweight;
			if (abs(*(*(*(nn + layer_no) + neuron_no) + weight_no)) > NN_MAX_WEIGHT)
				*(*(*(nn + layer_no) + neuron_no) + weight_no) -= random_dweight;
		}
	}
}
void NN_applyGamma(double ***gradient, double gamma)
{
	for (int layer_no = 2; layer_no < ***gradient; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < ***(gradient + layer_no); neuron_no++)
		{
			for (int weight_no = 2; weight_no < *(*(*(gradient + layer_no) + neuron_no)); weight_no++)
			{
				gradient[layer_no][neuron_no][weight_no] *= gamma;
			}
		}
	}
}
inline double NN_meanSquareError(double output, double expectedOutput)
{
	return (output - expectedOutput) * (output - expectedOutput);
}
inline double NN_crossEntropy(double SoftmaxOutput, double expectedOutput)
{
	if (expectedOutput > 0.9)
		return -log(SoftmaxOutput);
	return 0;
}
double (*NN_cost_funcs[])(double, double) = {NN_meanSquareError, NN_crossEntropy};
inline double NN_meanSquareErrorDerivative(double output, double expectedOutput)
{
	return 2 * (output - expectedOutput);
}
inline double NN_crossEntropyDerivative(double output, double expectedOutput)
{
	return output - expectedOutput;
}
double (*NN_costDerivativs_funcs[])(double, double) = {NN_meanSquareErrorDerivative, NN_crossEntropyDerivative};
double NN_cost(double ***nn, const double *expectedOutput)
{
	double cost = 0;
	double **outputLayer = nn[int(nn[0][0][0]) - 1];
	for (int neuron_no = 1; (neuron_no < outputLayer[0][0]) and (neuron_no < expectedOutput[0]); neuron_no++)
	{
		cost += NN_cost_funcs[int(nn[1][0][1])](outputLayer[neuron_no][1], expectedOutput[neuron_no]);
	}
	return cost;
}
void NN_derivate(double ***nn, double *expectedOutput, double ***gradient, double reward = 1) //Replace activation of each neuron of a neural net by d(Cost)/d(Activation of that neuron)
{
	//derivating output layer neurones w.r.t cost
	for (int neuron_no = 1; neuron_no < nn[int(nn[0][0][0]) - 1][0][0]; neuron_no++)
	{
		gradient[int(nn[0][0][0]) - 1][neuron_no][1] = NN_costDerivativs_funcs[int(nn[1][0][1])](nn[int(nn[0][0][0]) - 1][neuron_no][1], expectedOutput[neuron_no]);
	}
	//derivating rest of the layers neurones w.r.t cost
	for (int layer_no = int(nn[0][0][0]) - 2; layer_no > 1; layer_no--)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			gradient[layer_no][neuron_no][1] = 0;
			for (int nextLayer_neuron_no = 1; nextLayer_neuron_no < nn[layer_no + 1][0][0]; nextLayer_neuron_no++)
			{
				gradient[layer_no][neuron_no][1] += nn[layer_no + 1][nextLayer_neuron_no][neuron_no + 1] * gradient[layer_no + 1][nextLayer_neuron_no][1] * NN_aActivationDerivativef(nn[layer_no + 1][nextLayer_neuron_no][1], nn[layer_no + 1]); /*Using the formula activation of d(Cost)/d(Activation of neuron of layer no = L) = d(Cost)/d(Activation of neuron of layer no L-1)*d(Activation of neuron of layer no L-1)/d(Activation of neuron of layer no L)
= sigma{over all the neurons of next layer}(weight connecting a neuron of next layer*d(Cost)/d(Activation of that neuron of layer no L) * derivative of activation function)*/
			}
		}
	}
	//derivating weights
	for (int layer_no = 2; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			int weight_no = 2;
			for (; weight_no < nn[layer_no][neuron_no][0] - 1; weight_no++)
			{
				gradient[layer_no][neuron_no][weight_no] += reward * nn[layer_no - 1][weight_no - 1][1] * gradient[layer_no][neuron_no][1] * NN_aActivationDerivativef(nn[layer_no][neuron_no][1], nn[layer_no]);
			}
			gradient[layer_no][neuron_no][weight_no] += reward * gradient[layer_no][neuron_no][1] * NN_aActivationDerivativef(nn[layer_no][neuron_no][1], nn[layer_no]);
		}
	}
}
class NN_thread;
class NN_trainingData
{
	friend void NN_decendGradient(NN_thread *, NN_trainingData &, double);
	friend void NN_decendGradient(NN_thread *, NN_trainingData &, std::vector<double>, int, double);
	friend void NN_decendGradient(double ***, NN_trainingData &, double);
	friend void NN_decendGradient(double ***, NN_trainingData &, std::vector<double>, int, double);

  private:
	double ***gradient;
	int inserted = 0;
	int size;

  public:
	double **inputs;
	double **outputs;
	NN_trainingData(double ***nn, int arg_size)
	{
		size = arg_size;
		if (size > 0)
		{
			inputs = (double **)malloc(8 * size);
			outputs = (double **)malloc(8 * size);
			for (int example_no = 0; example_no < size; example_no++)
			{
				inputs[example_no] = (double *)malloc(8 * (NN_neuronCount(nn, 1) + 1));
				inputs[example_no][0] = NN_neuronCount(nn, 1) + 1;
				outputs[example_no] = (double *)malloc(8 * (NN_neuronCount(nn, NN_layerCount(nn)) + 1));
				outputs[example_no][0] = NN_neuronCount(nn, NN_layerCount(nn)) + 1;
			}
			gradient = NN_dublicate(nn);
			NN_clear(gradient);
		}
		else
		{
			printf("\nERROR: Size passed in NN_trainingData should be a positive integer. No memory was Allocated for NN_trainingData\nNN_trainingData::constructor() Aborted\n");
			exit(1);
		}
	}
	inline int get_size()
	{
		return size;
	}
	inline int get_insertionCount()
	{
		return inserted;
	}
	inline bool is_compleat()
	{
		return inserted >= size;
	}
	void insert(std::vector<double> input, std::vector<double> output)
	{
		if (is_compleat())
		{
			printf("\nWARNING: Cannot insert, NN_trainingData is Full. NOTE decleared size of NN_trainingData was- %i\nNN_trainingData::insert() Failed\n", get_size());
			return;
		}
		for (int neuron_no = 1; neuron_no < inputs[inserted][0]; neuron_no++)
			inputs[inserted][neuron_no] = input[neuron_no - 1];
		for (int neuron_no = 1; neuron_no < outputs[inserted][0]; neuron_no++)
			outputs[inserted][neuron_no] = output[neuron_no - 1];
		inserted++;
	}
	void clear()
	{
		inserted = 0;
	}
	~NN_trainingData()
	{
		if (size <= 0)
			return;
		for (int example_no = 0; example_no < size; example_no++)
		{
			free(inputs[example_no]);
			free(outputs[example_no]);
		}
		free(inputs);
		free(outputs);
		NN_delete(gradient);
	}
};
void NN_decendGradient(double ***nn, NN_trainingData &examples, double gamma = 0)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat NN_trainingData passed, decleared NN_trainingData size was %i but had only %i elements inserted\nNN_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	//computing gradient
	NN_applyGamma(examples.gradient, gamma);
	for (int example_no = 0; example_no < examples.get_size(); example_no++)
	{
		NN_input(nn, examples.inputs[example_no]);
		NN_process(nn);
		NN_derivate(nn, examples.outputs[example_no], examples.gradient);
	}
	//updating weights
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				nn[layer_no][neuron_no][weight_no] -= NN_LEARNING_RATE * examples.gradient[layer_no][neuron_no][weight_no] / examples.get_size();
			}
		}
	}
}
void NN_decendGradient(double ***nn, NN_trainingData &examples, std::vector<double> rewards, int totalRewards, double gamma = 0)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat NN_trainingData passed, decleared NN_trainingData size was %i but had only %i elements inserted\nNN_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	totalRewards += (-totalRewards + 1) * (totalRewards < 1);
	//computing gradient
	NN_applyGamma(examples.gradient, gamma);
	for (int example_no = 0; example_no < examples.get_size(); example_no++)
	{
		NN_input(nn, examples.inputs[example_no]);
		NN_process(nn);
		NN_derivate(nn, examples.outputs[example_no], examples.gradient, rewards[example_no]);
	}
	//updating weights
	for (int layer_no = 1; layer_no < nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < nn[layer_no][neuron_no][0]; weight_no++)
			{
				nn[layer_no][neuron_no][weight_no] -= NN_LEARNING_RATE * examples.gradient[layer_no][neuron_no][weight_no] / totalRewards;
			}
		}
	}
}
double NN_totalCost(double ***nn, NN_trainingData &examples)
{
	double totalCost = 0;
	for (int example_no = 0; example_no < examples.get_size(); example_no++)
	{
		NN_input(nn, examples.inputs[example_no]);
		NN_process(nn);
		totalCost += NN_cost(nn, examples.outputs[example_no]);
	}
	return totalCost / examples.get_size();
}
inline void pool_update_nns(int, NN_thread *);
void pool_executor(int, NN_thread *);
class NN_thread
{
	friend void pool_executor(int, NN_thread *);
	friend void pool_update_nns(int, NN_thread *);
	friend void pool_totalCost(int, NN_thread *);
	friend double NN_totalCost(NN_thread *, NN_trainingData &);
	friend void pool_decendGradient(int, NN_thread *);
	friend void NN_decendGradient(NN_thread *, NN_trainingData &, double);
	friend void NN_decendGradient(NN_thread *, NN_trainingData &, std::vector<double>, int, double);
	friend void pool_cmpt_activation_hidden(int, NN_thread *);
	friend void pool_cmpt_activation_output(int, NN_thread *);
	friend double **NN_process(NN_thread *);
	std::thread *threads;
	std::mutex locker;
	double ****nns;
	//shared memory between threads
	std::atomic<bool> alive = 0, invoking = 0;
	std::atomic<char> compleated = 0;
	int activeThread_ct = 0;
	void (*func)(int, NN_thread *);

	//functions for thread pool
	double **pool_input, **pool_output;
	double ****pool_gradient;
	double pool_totalCost_totalCost;
	double **pool_prev_layer, **pool_current_layer;

  public:
	double ***nn;
	int thread_ct;
	NN_thread(double ***arg_nn, int arg_thread_ct)
	{
		nn = arg_nn;
		if (arg_thread_ct <= 0)
		{
			printf("\nWARNING: Cannot run a program with 0 threads so thread_ct was set to 1\nNN_thread::constructor() Invalid argument\n");
			arg_thread_ct = 1;
		}
		thread_ct = arg_thread_ct;
		//Memory allocation
		nns = (double ****)malloc(8 * arg_thread_ct);
		nns[0] = (double ***)malloc(8);
		nns[0][0] = (double **)malloc(8);
		nns[0][0][0] = (double *)malloc(8);
		*nns[0][0][0] = arg_thread_ct;
		threads = new std::thread[thread_ct - 1];
		pool_input = (double **)malloc(8 * thread_ct);
		pool_output = (double **)malloc(8 * thread_ct);
		pool_gradient = (double ****)malloc(8 * thread_ct);
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			pool_gradient[thread_no] = NN_dublicate(nn);
			nns[thread_no] = NN_dublicate(nn);
			NN_clear(pool_gradient[thread_no]);
		}
	}
	void update_nns()
	{
		//Copying new values of nn to other nns
		invokeAll(pool_update_nns);
		joinAll();
	}
	bool create_threads()
	{
		if (alive)
			return false;
		else
			alive = 1;
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			threads[thread_no - 1] = std::thread(pool_executor, thread_no, this);
		}
		return true;
	}
	inline bool executed()
	{
		return (compleated == thread_ct - 1);
	}
	void joinAll()
	{
		while (!executed())
		{
		}
		return;
	}
	inline void input_cmpt_activation(int thread_no, double *neuron)
	{
		pool_input[thread_no] = neuron;
	}
	inline void input_decendGradient(int thread_no, double *input, double *output)
	{
		pool_input[thread_no] = input;
		pool_output[thread_no] = output;
	}
	inline void input_totalCost(int thread_no, double *input, double *output)
	{
		pool_input[thread_no] = input;
		pool_output[thread_no] = output;
	}
	void pool_applyGamma(double gamma)
	{
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			NN_applyGamma(pool_gradient[thread_no], gamma);
		}
	}
	inline void invokeAll(void (*arg_func)(int, NN_thread *))
	{
		activeThread_ct = thread_ct;
		func = arg_func;
		compleated = 0;
		invoking = invoking ^ 1;
	}
	inline void invoke(void (*arg_func)(int, NN_thread *), int arg_activeThread_ct)
	{
		activeThread_ct = arg_activeThread_ct;
		func = arg_func;
		compleated = 0;
		invoking = invoking ^ 1;
	}
	inline bool kill()
	{
		if (!alive)
			return false;
		alive = 0;
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			threads[thread_no - 1].join();
		}
		invoking = 0;
		return true;
	}
	~NN_thread()
	{
		kill();
		delete[] threads;
		free(pool_input);
		free(pool_output);
		for (int thread_no = 1; thread_no < thread_ct; thread_no++)
		{
			NN_delete(pool_gradient[thread_no]);
		}
		free(pool_gradient);
		for (int nns_no = 1; nns_no < ****nns; nns_no++)
		{
			NN_delete(nns[nns_no]);
		}
		free(nns[0][0][0]);
		free(nns[0][0]);
		free(nns[0]);
		free(nns);
	}
};
inline void pool_cmpt_activation_hidden(int thread_no, NN_thread *thread_ptr)
{
	double activation = 0;
	int neuron_no = 1;
	//aliasing neuron == pool_input[thread_no]; because pool_input stores neurons
	for (; neuron_no < **thread_ptr->pool_prev_layer; neuron_no++)
	{
		activation += *(*(thread_ptr->pool_prev_layer + neuron_no) + 1) * (*((thread_ptr->pool_input)[thread_no] + neuron_no + 1)); //Adding weigted sum
	}
	activation += *((thread_ptr->pool_input)[thread_no] + neuron_no + 1); //Adding bias
	*((thread_ptr->pool_input)[thread_no] + 1) = NN_activationf(activation, thread_ptr->pool_current_layer);
}
inline void pool_decendGradient(int thread_no, NN_thread *thread_ptr)
{
	NN_input((thread_ptr->nns)[thread_no], (thread_ptr->pool_input)[thread_no]);
	NN_process((thread_ptr->nns)[thread_no]);
	NN_derivate((thread_ptr->nns)[thread_no], (thread_ptr->pool_output)[thread_no], (thread_ptr->pool_gradient)[thread_no]);
}
inline void pool_totalCost(int thread_no, NN_thread *thread_ptr)
{
	NN_input((thread_ptr->nns)[thread_no], (thread_ptr->pool_input)[thread_no]);
	NN_process((thread_ptr->nns)[thread_no]);
	double cost = NN_cost((thread_ptr->nns)[thread_no], (thread_ptr->pool_output)[thread_no]);
	thread_ptr->locker.lock();
	thread_ptr->pool_totalCost_totalCost += cost;
	thread_ptr->locker.unlock();
}
inline void pool_update_nns(int thread_no, NN_thread *thread_ptr)
{
	NN_copy(thread_ptr->nns[thread_no], thread_ptr->nn);
}
void pool_executor(int thread_no, NN_thread *thread_ptr)
{
	bool local_invoking = 0;
	while (1)
	{
		while (local_invoking == thread_ptr->invoking)
		{
			if (!thread_ptr->alive)
				return;
		}
		local_invoking = thread_ptr->invoking; //deactivating

		if (thread_no < thread_ptr->activeThread_ct)
			thread_ptr->func(thread_no, thread_ptr);

		thread_ptr->compleated++;
	}
}
double **NN_process(NN_thread *thread_ptr)
{
	int layer_no = 2;
	for (; layer_no < ***thread_ptr->nn; layer_no++)
	{
		thread_ptr->pool_prev_layer = *(thread_ptr->nn + layer_no - 1);
		thread_ptr->pool_current_layer = *(thread_ptr->nn + layer_no);
		int neuron_no = 1;
		while (neuron_no + thread_ptr->thread_ct <= ***(thread_ptr->nn + layer_no))
		{
			for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
			{
				thread_ptr->input_cmpt_activation(thread_no, *(*(thread_ptr->nn + layer_no) + neuron_no + thread_no));
			}
			thread_ptr->invokeAll(pool_cmpt_activation_hidden);
			NN_cmpt_activation(*(thread_ptr->nn + layer_no - 1), *(thread_ptr->nn + layer_no), *(*(thread_ptr->nn + layer_no) + neuron_no));
			thread_ptr->joinAll();
			neuron_no += thread_ptr->thread_ct;
		}
		int thread_no = 1;
		for (; neuron_no < ***(thread_ptr->nn + layer_no); neuron_no++)
		{
			thread_ptr->input_cmpt_activation(thread_no, *(*(thread_ptr->nn + layer_no) + neuron_no));
			thread_no++;
		}
		thread_ptr->invoke(pool_cmpt_activation_hidden, thread_no);
		thread_ptr->joinAll();
	}
	if (thread_ptr->nn[layer_no - 1][0][1] == NN_type_Softmax)
		NN_SoftmaxManager(thread_ptr->nn[layer_no - 1]);
	return *(thread_ptr->nn + (int)(***thread_ptr->nn) - 1);
}
void NN_decendGradient(NN_thread *thread_ptr, NN_trainingData &examples, double gamma = 0)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat NN_trainingData passed, decleared NN_trainingData size was %i but had only %i elements inserted\nNN_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	//computing gradient
	NN_applyGamma(examples.gradient, gamma);
	thread_ptr->pool_applyGamma(gamma);
	int example_no = 0;
	while (example_no + thread_ptr->thread_ct <= examples.get_size())
	{
		for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
		{
			thread_ptr->input_decendGradient(thread_no, examples.inputs[example_no + thread_no], examples.outputs[example_no + thread_no]);
		}
		thread_ptr->invokeAll(pool_decendGradient);
		NN_input(thread_ptr->nn, examples.inputs[example_no]);
		NN_process(thread_ptr->nn);
		NN_derivate(thread_ptr->nn, examples.outputs[example_no], examples.gradient);
		thread_ptr->joinAll();
		example_no += thread_ptr->thread_ct;
	}
	int thread_no = 1;
	for (; example_no < examples.get_size(); example_no++)
	{
		thread_ptr->input_decendGradient(thread_no, examples.inputs[example_no], examples.outputs[example_no]);
		thread_no++;
	}
	thread_ptr->invoke(pool_decendGradient, thread_no);
	thread_ptr->joinAll();
	//updating weights
	for (int layer_no = 1; layer_no < thread_ptr->nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < thread_ptr->nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < thread_ptr->nn[layer_no][neuron_no][0]; weight_no++)
			{
				double weight_update = examples.gradient[layer_no][neuron_no][weight_no];
				for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
					weight_update += thread_ptr->pool_gradient[thread_no][layer_no][neuron_no][weight_no];

				thread_ptr->nn[layer_no][neuron_no][weight_no] -= NN_LEARNING_RATE * weight_update / examples.get_size();
			}
		}
	}
}
void NN_decendGradient(NN_thread *thread_ptr, NN_trainingData &examples, std::vector<double> rewards, int totalRewards, double gamma = 0)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat NN_trainingData passed, decleared NN_trainingData size was %i but had only %i elements inserted\nNN_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return;
	}
	//computing gradient
	NN_applyGamma(examples.gradient, gamma);
	thread_ptr->pool_applyGamma(gamma);
	int example_no = 0;
	while (example_no + thread_ptr->thread_ct <= examples.get_size())
	{
		for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
		{
			thread_ptr->input_decendGradient(thread_no, examples.inputs[example_no + thread_no], examples.outputs[example_no + thread_no]);
		}
		thread_ptr->invokeAll(pool_decendGradient);
		NN_input(thread_ptr->nn, examples.inputs[example_no]);
		NN_process(thread_ptr->nn);
		NN_derivate(thread_ptr->nn, examples.outputs[example_no], examples.gradient);
		thread_ptr->joinAll();
		example_no += thread_ptr->thread_ct;
	}
	int thread_no = 1;
	for (; example_no < examples.get_size(); example_no++)
	{
		thread_ptr->input_decendGradient(thread_no, examples.inputs[example_no], examples.outputs[example_no]);
		thread_no++;
	}
	thread_ptr->invoke(pool_decendGradient, thread_no);
	thread_ptr->joinAll();
	//updating weights
	for (int layer_no = 1; layer_no < thread_ptr->nn[0][0][0]; layer_no++)
	{
		for (int neuron_no = 1; neuron_no < thread_ptr->nn[layer_no][0][0]; neuron_no++)
		{
			for (int weight_no = 2; weight_no < thread_ptr->nn[layer_no][neuron_no][0]; weight_no++)
			{
				double weight_update = examples.gradient[layer_no][neuron_no][weight_no];
				for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
					weight_update += thread_ptr->pool_gradient[thread_no][layer_no][neuron_no][weight_no];

				thread_ptr->nn[layer_no][neuron_no][weight_no] -= NN_LEARNING_RATE * weight_update / examples.get_size();
			}
		}
	}
}
double NN_totalCost(NN_thread *thread_ptr, NN_trainingData &examples)
{
	if (!examples.is_compleat())
	{
		printf("\nERROR: Incompleat NN_trainingData passed, decleared NN_trainingData size was %i but had only %i elements inserted\nNN_decendGradient() Aborted\n", examples.get_size(), examples.get_insertionCount());
		exit(1);
		return -1;
	}
	double cost;
	thread_ptr->pool_totalCost_totalCost = 0;
	int example_no = 0;
	while (example_no + thread_ptr->thread_ct <= examples.get_size())
	{
		for (int thread_no = 1; thread_no < thread_ptr->thread_ct; thread_no++)
		{
			thread_ptr->input_totalCost(thread_no, examples.inputs[example_no + thread_no], examples.outputs[example_no + thread_no]);
		}
		thread_ptr->invokeAll(pool_totalCost);
		NN_input(thread_ptr->nn, examples.inputs[example_no]);
		NN_process(thread_ptr->nn);
		cost = NN_cost(thread_ptr->nn, examples.outputs[example_no]);
		thread_ptr->joinAll();
		thread_ptr->pool_totalCost_totalCost += cost;
		example_no += thread_ptr->thread_ct;
	}
	int thread_no = 1;
	for (; example_no < examples.get_size(); example_no++)
	{
		thread_ptr->input_totalCost(thread_no, examples.inputs[example_no], examples.outputs[example_no]);
		thread_no++;
	}
	thread_ptr->invoke(pool_totalCost, thread_no);
	thread_ptr->joinAll();
	return thread_ptr->pool_totalCost_totalCost / examples.get_size();
}
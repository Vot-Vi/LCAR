#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

typedef struct {
    int batch_size;
    float *z;
	 float *grad;

} DenseCache;
typedef struct {
	int input_size;
   int output_size;
// FORWARD
   float *weights; // i,j donde i es la neurona y j es el input
   float *bias; // i, hay uno por neurona
	float *input; //Si no soy la primera neurona, esto se puede asignar a la anterior
	float *output;
// BACKWARD
   float *grad_weights; // i,j donde i son los input y j es la neurona
   float *grad_bias; // j, hay uno por neurona
	int initialized;
   DenseCache cache;
} DenseLayer;
typedef struct {
   int num_layers;
   DenseLayer *layers;
} NeuralNetwork;

float RELU(float x){
	if (x>0) return x;
	else return 0;
}

float RELU_derivative(float x){
	if (x>0) return 1;
	else return 0;
}

int NN_init(NeuralNetwork* neuralnet,int num_layers){
	if (neuralnet == NULL){
		printf("No se inicializo la NN\n");
		return -1;
	}
	neuralnet->num_layers = num_layers;
	DenseLayer* Layer = (DenseLayer*)malloc(num_layers*sizeof(DenseLayer));
	if (Layer == NULL){
		printf("No se inicializo la NN\n");
		return -1;
	}
	for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
		Layer[layer_indx].initialized = 0;
	}
	neuralnet->layers = Layer;
	return 0;
}

int NN_Layer_innit(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size){
	if (layer_indx !=0 && input_size != neuralnet->layers[layer_indx-1].output_size){
		printf("El tamaño del output anterior no es igual a el input de esta capa \n");
		return -1;
	}
	if (layer_indx>neuralnet->num_layers-1){
		printf("El indice de layer  %d es mayor que la capacidad inicializada de la nn\n",layer_indx);
	}
	if (neuralnet->layers[layer_indx].initialized!=0){
		printf("%d ",layer_indx);
		printf("Esta capa ya esta inicializada compadre \n");
		return -1;
	}
	DenseLayer* Layer = &(neuralnet->layers[layer_indx]);
	Layer->input_size = input_size;
	Layer->output_size = output_size;
	Layer->weights = (float*)malloc(input_size*output_size*sizeof(float));
	Layer->bias    = (float*)malloc(output_size*sizeof(float));
	Layer->grad_weights = (float*)malloc(input_size*output_size*sizeof(float));
	Layer->grad_bias    = (float*)malloc(output_size*sizeof(float));
	Layer->cache.grad = (float*)malloc(output_size*sizeof(float));
	Layer->cache.z = (float*)malloc(output_size*sizeof(float));
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			Layer->weights[i*input_size+j] = (float)drand48()-0.5; //random entre (-0.5,0.5)
		}
		Layer->bias[i] = 0.0f;
	}
	neuralnet->layers[layer_indx].initialized=1;
	return 0;
}
void  NN_print_layer(NeuralNetwork* neuralnet,int layer_indx){
	DenseLayer layer=neuralnet->layers[layer_indx];
	int output_size = layer.output_size;
	int input_size= layer.input_size;
	printf("Printing layer %d weights W:\n [\n",layer_indx);
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			printf("%f ",layer.weights[i*input_size+j]);
		}
		printf("\n");
	}
	printf("]\n");
	printf("Printing layer %d bias B:\n [\n",layer_indx);
	for (int i = 0; i < output_size; i++){
		printf("%f\n",layer.bias[i]);
	}
	printf("]\n");
}
int NN_loop(NeuralNetwork* neuralnet,int epochs,float *X,float *Y,int n_examples){
	int num_layers = neuralnet->num_layers;
	int final_size=neuralnet->layers[num_layers-1].output_size;
	float* loss = (float*)malloc(final_size*sizeof(float));
	float* grad_buffer = (float*)malloc(final_size*sizeof(float));

	for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
		int input_size = neuralnet->layers[layer_indx].input_size;
		int output_size = neuralnet->layers[layer_indx].output_size;
		if (layer_indx == 0){
			neuralnet->layers[layer_indx].output= (float*)calloc(output_size,sizeof(float));
		}
		else{
			neuralnet->layers[layer_indx].input = neuralnet->layers[layer_indx-1].output;
			neuralnet->layers[layer_indx].output=  (float*)calloc(output_size,sizeof(float));
		}
	}
	for (int ep = 0; ep < epochs; ep++){
		//FORWARD
		float loss_avg = 0;
		for (int ex=0; ex < n_examples; ex++){
			float* grad = grad_buffer;
			float* x = X+ex*neuralnet->layers[0].input_size;
			float* y = Y+ex*neuralnet->layers[num_layers-1].output_size;
			neuralnet->layers[0].input = x;
			for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
				int input_size = neuralnet->layers[layer_indx].input_size;
				int output_size = neuralnet->layers[layer_indx].output_size;
				for (int i = 0; i < output_size; i++){
					neuralnet->layers[layer_indx].output[i]=0;
					for (int j = 0; j < input_size; j++){
						neuralnet->layers[layer_indx].output[i]+=	neuralnet->layers[layer_indx].weights[i*input_size+j] * neuralnet->layers[layer_indx].input[j];
					}
					neuralnet->layers[layer_indx].output[i]+=neuralnet->layers[layer_indx].bias[i];
					neuralnet->layers[layer_indx].cache.z[i] = neuralnet->layers[layer_indx].output[i];
					neuralnet->layers[layer_indx].output[i] = RELU(neuralnet->layers[layer_indx].output[i]);
				}
			}
			//BACKWARD
			for (int i = 0; i < final_size; i++){
				loss[i] = 0.5*(neuralnet->layers[num_layers-1].output[i] - y[i])*(neuralnet->layers[num_layers-1].output[i] - y[i]);
				grad[i] = (neuralnet->layers[num_layers-1].output[i] - y[i]);
				loss_avg += (loss[i]/final_size)/n_examples;
				//printf("La red da %f y la respuesta es %f loss = %f\n",neuralnet->layers[num_layers-1].output[i],y[i],loss[i]);
			}
			for (int layer_indx =num_layers-1;layer_indx >= 0;layer_indx--){
				int input_size = neuralnet->layers[layer_indx].input_size;
				int output_size = neuralnet->layers[layer_indx].output_size;
				for (int i = 0; i < output_size; i++){ 
					float grad_z = grad[i]; 
					grad_z*=RELU_derivative(neuralnet->layers[layer_indx].cache.z[i]);
					neuralnet->layers[layer_indx].grad_bias[i]=grad_z;
					// No olvidar que cuando tenga funcion de activacion
					// tengo que hacer grad_z*d funcion activacion
					for (int j = 0; j < input_size; j++){
						neuralnet->layers[layer_indx].grad_weights[i*input_size+j] = grad_z*neuralnet->layers[layer_indx].input[j];
					}
				}
				if ( layer_indx > 0){
					for (int j = 0; j < input_size; j++){
						float sum = 0;
						for (int i = 0; i < output_size; i++){
							sum+= grad[i]*neuralnet->layers[layer_indx].weights[i*input_size+j];	
						}
						neuralnet->layers[layer_indx-1].cache.grad[j] = sum;
					}
					grad = neuralnet->layers[layer_indx-1].cache.grad;
				}
			}
			float learning_rate = 0.01;
			for (int layer_indx = 0; layer_indx < num_layers; layer_indx++) {
				int input_size = neuralnet->layers[layer_indx].input_size;
				int output_size = neuralnet->layers[layer_indx].output_size;
				for (int i = 0; i < output_size; i++) {
					neuralnet->layers[layer_indx].bias[i]-=learning_rate*neuralnet->layers[layer_indx].grad_bias[i];
					for (int j = 0; j < input_size; j++) {
						neuralnet->layers[layer_indx].weights[i*input_size+j]-=learning_rate*neuralnet->layers[layer_indx].grad_weights[i*input_size+j];
					}
				}
			}
		}
		printf("Epoch %d, Loss: %f\n", ep, loss_avg);
	}	

	return 0;
}
int main(){
	NeuralNetwork* neuralnet = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	NN_init(neuralnet,4);
	int input_size = 1;
	int n_examples = 100;
	int out_size = 1;
	float *X = (float*)malloc(input_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) X[i]=drand48();
	float *Y = (float*)malloc(out_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) Y[i]=2*X[i]*X[i]+1;
	NN_Layer_innit(neuralnet,0,input_size,8);
	NN_Layer_innit(neuralnet,1,8,8);
	NN_Layer_innit(neuralnet,2,8,8);
	NN_Layer_innit(neuralnet,3,8,out_size);
	//NN_print_layer(neuralnet,0);
	//NN_print_layer(neuralnet,1);
	double start = omp_get_wtime();
	NN_loop(neuralnet,100,X,Y,n_examples);
	double end = omp_get_wtime();
	printf("Esto tardo %lf\n",end-start);
	return 0;
}

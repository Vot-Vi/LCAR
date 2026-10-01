#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

//ReLU
float RELU(float x){
	if (x>0) return x;
	else return 0;
}

float RELU_derivative(float x){
	if (x>0) return 1;
	else return 0;
}
//Fin ReLU

//Sigmoide
float Sigmoide(float x){
	return 1/(1+exp(-x));
}

float Sigmoide_derivative(float x){
	return Sigmoide(x)*(1-Sigmoide(x));
}
//Fin Sigmoide

//Tanh
float Tanh(float x){
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

float Tanh_derivative(float x){
	return 1-Tanh(x)*Tanh(x);
}
//Fin Tanh

//SGD
void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

void random_permutation(int *indices, int n)
{
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }

    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap(&indices[i], &indices[j]);
    }
}

//Fin SGD

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

int NN_Layer_innit(NeuralNetwork* neuralnet, int layer_indx, int input_size,int output_size, int actfunction){
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
	Layer->bias    = (float*)malloc(output_size*sizeof(float));
	Layer->grad_weights = (float*)malloc(input_size*output_size*sizeof(float));
	Layer->grad_bias    = (float*)malloc(output_size*sizeof(float));
	Layer->cache.grad = (float*)malloc(output_size*sizeof(float));
	Layer->cache.z = (float*)malloc(output_size*sizeof(float));
	Layer->weights = (float*)malloc(input_size*output_size*sizeof(float));
	

	//Revisar un solo if
	float ab = 0;
	if (actfunction == 1){
		ab = sqrtf(6/(float)input_size);
	}
	if (actfunction == 2 || actfunction == 3){
		ab = sqrtf(6/(float)(input_size + output_size));
	}
			
	for (int i = 0; i < output_size; i++){
		for (int j = 0; j < input_size; j++){
			float rand = (float)drand48();
			Layer->weights[i*input_size+j] = 2*ab*rand - ab; //random entre (-ab,ab)
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

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y, int n_examples, int actfunction, int batch_size, float learning_rate){

    int num_layers = neuralnet->num_layers;

    int final_size = neuralnet->layers[num_layers - 1].output_size;

    float *loss = (float *)malloc(final_size * sizeof(float));

    float *grad_buffer = (float *)malloc(final_size * sizeof(float));

    /*
        Índices para hacer shuffle
    */
    int *indices = (int *)malloc(n_examples * sizeof(int));


    /*
        Reservamos los outputs de cada capa
    */
    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
        int output_size = neuralnet->layers[layer_indx].output_size;

        neuralnet->layers[layer_indx].output = (float *)calloc(output_size, sizeof(float));

        /*
            La primera capa recibe X directamente.
            Las siguientes reciben el output de la capa anterior.
        */
        if (layer_indx > 0) { 
            neuralnet->layers[layer_indx].input = neuralnet->layers[layer_indx - 1].output;
        }
    }


    /*
        ============================
        INICIO DEL ENTRENAMIENTO
        ============================
    */
    if (actfunction == 1) {
        for (int ep = 0; ep < epochs; ep++)
        {
            /*
                Shuffle de los ejemplos
            */
            random_permutation(indices, n_examples);

            float loss_avg = 0.0f;


            /*
                ===================================
                RECORREMOS LOS MINIBATCHES
                ===================================
            */
       
            for (int batch_start = 0; batch_start < n_examples; batch_start += batch_size){

                /*
                    Tamaño real del batch.
                    Importante para el último minibatch.
                */
                int current_batch_size = batch_size;

                if (batch_start + batch_size > n_examples) {
                    current_batch_size = n_examples - batch_start;
                }


                /*
                    ===================================
                    PONER LOS GRADIENTES EN CERO
                    ===================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++) {
                        neuralnet->layers[layer_indx] .grad_bias[i] = 0.0f;
                    }


                    for (int i = 0; i < output_size; i++) {
                        for (int j = 0; j < input_size; j++) {
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] = 0.0f;
                        }
                    }
                }


                /*
                    ===================================
                    RECORREMOS LOS EJEMPLOS DEL BATCH
                    ===================================
                */

                for (int b = 0; b < current_batch_size; b++){

                    /*
                        Índice real del ejemplo
                    */
                    int ex = indices[batch_start + b];


                    /*
                        --------------------------------
                        INPUT Y TARGET
                        --------------------------------
                    */

                    float *x = X + ex * neuralnet->layers[0].input_size;

                    float *y = Y + ex * neuralnet->layers[num_layers - 1].output_size;

                    neuralnet->layers[0].input = x;


                    /*
                        ===================================
                        FORWARD
                        ===================================
                    */

                    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        for (int i = 0; i < output_size; i++){

                            neuralnet->layers[layer_indx].output[i] = 0.0f;


                            /*
                                W*x
                            */
                            for (int j = 0; j < input_size; j++){
                                neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].weights[i * input_size + j] * neuralnet->layers[layer_indx].input[j];
                            }


                            /*
                                + bias
                            */
                            neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].bias[i];


                            /*
                                Guardamos z
                            */
                            neuralnet->layers[layer_indx].cache.z[i] = neuralnet->layers[layer_indx].output[i];


                            /*
                                Activación
                            */
                            neuralnet->layers[layer_indx].output[i] = RELU(neuralnet->layers[layer_indx].output[i]);
                        }
                    }


                    /*
                        ===================================
                        LOSS
                        ===================================
                    */

                    for (int i = 0; i < final_size; i++){
                        float error = neuralnet->layers[num_layers - 1].output[i] - y[i];


                        loss[i] = 0.5f * error * error;


                        /*
                            dL/doutput
                        */
                        grad_buffer[i] = error;


                        loss_avg += (loss[i] / final_size) / n_examples;
                    }


                    /*
                        ===================================
                        BACKWARD
                        ===================================
                    */

                    float *grad = grad_buffer;


                    for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--){

                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        /*
                            --------------------------------
                            dL/dz
                            --------------------------------
                        */

                        for (int i = 0; i < output_size; i++){

                            float grad_z = grad[i];


                            /*
                                Derivada de activación
                            */
                            grad_z *= RELU_derivative(neuralnet->layers[layer_indx].cache.z[i]);


                            /*
                                =================================
                                ACUMULAMOS grad_bias
                                =================================
                            */

                            neuralnet->layers[layer_indx].grad_bias[i] += grad_z;


                            /*
                                =================================
                                ACUMULAMOS grad_weights
                                =================================
                            */

                            for (int j = 0; j < input_size; j++) {
                                neuralnet->layers[layer_indx].grad_weights[i * input_size + j] += grad_z*neuralnet->layers[layer_indx].input[j];
                            }
                        }


                        /*
                            =================================
                            PROPAGAR GRADIENTE HACIA ATRÁS
                            =================================
                        */

                        if (layer_indx > 0){

                            for (int j = 0; j < input_size; j++){

                                float sum = 0.0f;


                                for (int i = 0; i < output_size; i++){
                                    sum += grad[i]*neuralnet->layers[layer_indx].weights[i * input_size + j];
                                }


                                neuralnet->layers[layer_indx - 1].cache.grad[j]= sum;
                            }


                            grad = neuralnet->layers[layer_indx - 1].cache.grad;
                        }
                    }
                }


                /*
                    =========================================
                    PROMEDIAR LOS GRADIENTES DEL MINIBATCH
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
                {

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    /*
                        grad_bias /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        neuralnet->layers[layer_indx].grad_bias[i] /= current_batch_size;
                    }


                    /*
                        grad_weights /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] /= current_batch_size;
                        }
                    }
                }


                /*
                    =========================================
                    ACTUALIZAR PESOS
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
                {

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++)
                    {

                        /*
                            bias -= learning_rate * grad_bias
                        */

                        neuralnet->layers[layer_indx].bias[i] -= learning_rate * neuralnet->layers[layer_indx].grad_bias[i];


                        /*
                            weights -= learning_rate * grad_weights
                        */

                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].weights[i * input_size + j]-=learning_rate*neuralnet->layers[layer_indx].grad_weights[i * input_size + j];
                        }
                    }
                }
            }
        printf("Epoch %d, Loss: %f\n", ep, loss_avg);
        }
    }

    if (actfunction == 2) {
        for (int ep = 0; ep < epochs; ep++)
        {
            /*
                Shuffle de los ejemplos
            */
            random_permutation(indices, n_examples);

            float loss_avg = 0.0f;


            /*
                ===================================
                RECORREMOS LOS MINIBATCHES
                ===================================
            */
       
            for (int batch_start = 0; batch_start < n_examples; batch_start += batch_size){

                /*
                    Tamaño real del batch.
                    Importante para el último minibatch.
                */
                int current_batch_size = batch_size;

                if (batch_start + batch_size > n_examples) {
                    current_batch_size = n_examples - batch_start;
                }


                /*
                    ===================================
                    PONER LOS GRADIENTES EN CERO
                    ===================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++) {
                        neuralnet->layers[layer_indx] .grad_bias[i] = 0.0f;
                    }


                    for (int i = 0; i < output_size; i++) {
                        for (int j = 0; j < input_size; j++) {
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] = 0.0f;
                        }
                    }
                }


                /*
                    ===================================
                    RECORREMOS LOS EJEMPLOS DEL BATCH
                    ===================================
                */

                for (int b = 0; b < current_batch_size; b++){

                    /*
                        Índice real del ejemplo
                    */
                    int ex = indices[batch_start + b];


                    /*
                        --------------------------------
                        INPUT Y TARGET
                        --------------------------------
                    */

                    float *x = X + ex * neuralnet->layers[0].input_size;

                    float *y = Y + ex * neuralnet->layers[num_layers - 1].output_size;

                    neuralnet->layers[0].input = x;


                    /*
                        ===================================
                        FORWARD
                        ===================================
                    */

                    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        for (int i = 0; i < output_size; i++){

                            neuralnet->layers[layer_indx].output[i] = 0.0f;


                            /*
                                W*x
                            */
                            for (int j = 0; j < input_size; j++){
                                neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].weights[i * input_size + j] * neuralnet->layers[layer_indx].input[j];
                            }


                            /*
                                + bias
                            */
                            neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].bias[i];


                            /*
                                Guardamos z
                            */
                            neuralnet->layers[layer_indx].cache.z[i] = neuralnet->layers[layer_indx].output[i];


                            /*
                                Activación
                            */
                            neuralnet->layers[layer_indx].output[i] = Sigmoide(neuralnet->layers[layer_indx].output[i]);
                        }
                    }


                    /*
                        ===================================
                        LOSS
                        ===================================
                    */

                    for (int i = 0; i < final_size; i++){
                        float error = neuralnet->layers[num_layers - 1].output[i] - y[i];


                        loss[i] = 0.5f * error * error;


                        /*
                            dL/doutput
                        */
                        grad_buffer[i] = error;


                        loss_avg += (loss[i] / final_size) / n_examples;
                    }


                    /*
                        ===================================
                        BACKWARD
                        ===================================
                    */

                    float *grad = grad_buffer;


                    for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--){

                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        /*
                            --------------------------------
                            dL/dz
                            --------------------------------
                        */

                        for (int i = 0; i < output_size; i++){

                            float grad_z = grad[i];


                            /*
                                Derivada de activación
                            */
                            grad_z *= Sigmoide_derivative(neuralnet->layers[layer_indx].cache.z[i]);


                            /*
                                =================================
                                ACUMULAMOS grad_bias
                                =================================
                            */

                            neuralnet->layers[layer_indx].grad_bias[i] += grad_z;


                            /*
                                =================================
                                ACUMULAMOS grad_weights
                                =================================
                            */

                            for (int j = 0; j < input_size; j++) {
                                neuralnet->layers[layer_indx].grad_weights[i * input_size + j] += grad_z*neuralnet->layers[layer_indx].input[j];
                            }
                        }


                        /*
                            =================================
                            PROPAGAR GRADIENTE HACIA ATRÁS
                            =================================
                        */

                        if (layer_indx > 0){

                            for (int j = 0; j < input_size; j++){

                                float sum = 0.0f;


                                for (int i = 0; i < output_size; i++){
                                    sum += grad[i]*neuralnet->layers[layer_indx].weights[i * input_size + j];
                                }


                                neuralnet->layers[layer_indx - 1].cache.grad[j]= sum;
                            }


                            grad = neuralnet->layers[layer_indx - 1].cache.grad;
                        }
                    }
                }


                /*
                    =========================================
                    PROMEDIAR LOS GRADIENTES DEL MINIBATCH
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
                {

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    /*
                        grad_bias /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        neuralnet->layers[layer_indx].grad_bias[i] /= current_batch_size;
                    }


                    /*
                        grad_weights /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] /= current_batch_size;
                        }
                    }
                }


                /*
                    =========================================
                    ACTUALIZAR PESOS
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
                {

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++)
                    {

                        /*
                            bias -= learning_rate * grad_bias
                        */

                        neuralnet->layers[layer_indx].bias[i] -= learning_rate * neuralnet->layers[layer_indx].grad_bias[i];


                        /*
                            weights -= learning_rate * grad_weights
                        */

                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].weights[i * input_size + j]-=learning_rate*neuralnet->layers[layer_indx].grad_weights[i * input_size + j];
                        }
                    }
                }
            }
        printf("Epoch %d, Loss: %f\n", ep, loss_avg);
        }
    }
    if (actfunction == 3) {
        for (int ep = 0; ep < epochs; ep++){
            /*
                Shuffle de los ejemplos
            */
            random_permutation(indices, n_examples);

            float loss_avg = 0.0f;


            /*
                ===================================
                RECORREMOS LOS MINIBATCHES
                ===================================
            */
       
            for (int batch_start = 0; batch_start < n_examples; batch_start += batch_size){

                /*
                    Tamaño real del batch.
                    Importante para el último minibatch.
                */
                int current_batch_size = batch_size;

                if (batch_start + batch_size > n_examples) {
                    current_batch_size = n_examples - batch_start;
                }


                /*
                    ===================================
                    PONER LOS GRADIENTES EN CERO
                    ===================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++) {
                        neuralnet->layers[layer_indx] .grad_bias[i] = 0.0f;
                    }


                    for (int i = 0; i < output_size; i++) {
                        for (int j = 0; j < input_size; j++) {
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] = 0.0f;
                        }
                    }
                }


                /*
                    ===================================
                    RECORREMOS LOS EJEMPLOS DEL BATCH
                    ===================================
                */

                for (int b = 0; b < current_batch_size; b++){

                    /*
                        Índice real del ejemplo
                    */
                    int ex = indices[batch_start + b];


                    /*
                        --------------------------------
                        INPUT Y TARGET
                        --------------------------------
                    */

                    float *x = X + ex * neuralnet->layers[0].input_size;

                    float *y = Y + ex * neuralnet->layers[num_layers - 1].output_size;

                    neuralnet->layers[0].input = x;


                    /*
                        ===================================
                        FORWARD
                        ===================================
                    */

                    for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        for (int i = 0; i < output_size; i++){

                            neuralnet->layers[layer_indx].output[i] = 0.0f;


                            /*
                                W*x
                            */
                            for (int j = 0; j < input_size; j++){
                                neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].weights[i * input_size + j] * neuralnet->layers[layer_indx].input[j];
                            }


                            /*
                                + bias
                            */
                            neuralnet->layers[layer_indx].output[i] += neuralnet->layers[layer_indx].bias[i];


                            /*
                                Guardamos z
                            */
                            neuralnet->layers[layer_indx].cache.z[i] = neuralnet->layers[layer_indx].output[i];


                            /*
                                Activación
                            */
                            neuralnet->layers[layer_indx].output[i] = Tanh(neuralnet->layers[layer_indx].output[i]);
                        }
                    }


                    /*
                        ===================================
                        LOSS
                        ===================================
                    */

                    for (int i = 0; i < final_size; i++){
                        float error = neuralnet->layers[num_layers - 1].output[i] - y[i];


                        loss[i] = 0.5f * error * error;


                        /*
                            dL/doutput
                        */
                        grad_buffer[i] = error;


                        loss_avg += (loss[i] / final_size) / n_examples;
                    }


                    /*
                        ===================================
                        BACKWARD
                        ===================================
                    */

                    float *grad = grad_buffer;


                    for (int layer_indx = num_layers - 1; layer_indx >= 0; layer_indx--){

                        int input_size = neuralnet->layers[layer_indx].input_size;

                        int output_size = neuralnet->layers[layer_indx].output_size;


                        /*
                            --------------------------------
                            dL/dz
                            --------------------------------
                        */

                        for (int i = 0; i < output_size; i++){

                            float grad_z = grad[i];


                            /*
                                Derivada de activación
                            */

                            grad_z *= Tanh_derivative(neuralnet->layers[layer_indx].cache.z[i]);


                            /*
                                =================================
                                ACUMULAMOS grad_bias
                                =================================
                            */

                            neuralnet->layers[layer_indx].grad_bias[i] += grad_z;


                            /*
                                =================================
                                ACUMULAMOS grad_weights
                                =================================
                            */

                            for (int j = 0; j < input_size; j++) {
                                neuralnet->layers[layer_indx].grad_weights[i * input_size + j] += grad_z*neuralnet->layers[layer_indx].input[j];
                            }
                        }


                        /*
                            =================================
                            PROPAGAR GRADIENTE HACIA ATRÁS
                            =================================
                        */

                        if (layer_indx > 0){

                            for (int j = 0; j < input_size; j++){

                                float sum = 0.0f;


                                for (int i = 0; i < output_size; i++){
                                    sum += grad[i]*neuralnet->layers[layer_indx].weights[i * input_size + j];
                                }


                                neuralnet->layers[layer_indx - 1].cache.grad[j]= sum;
                            }


                            grad = neuralnet->layers[layer_indx - 1].cache.grad;
                        }
                    }
                }


                /*
                    =========================================
                    PROMEDIAR LOS GRADIENTES DEL MINIBATCH
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    /*
                        grad_bias /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        neuralnet->layers[layer_indx].grad_bias[i] /= current_batch_size;
                    }


                    /*
                        grad_weights /= batch_size
                    */
                    for (int i = 0; i < output_size; i++)
                    {
                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].grad_weights[i * input_size + j] /= current_batch_size;
                        }
                    }
                }


                /*
                    =========================================
                    ACTUALIZAR PESOS
                    =========================================
                */

                for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){

                    int input_size = neuralnet->layers[layer_indx].input_size;

                    int output_size = neuralnet->layers[layer_indx].output_size;


                    for (int i = 0; i < output_size; i++){

                        /*
                            bias -= learning_rate * grad_bias
                        */

                        neuralnet->layers[layer_indx].bias[i] -= learning_rate * neuralnet->layers[layer_indx].grad_bias[i];


                        /*
                            weights -= learning_rate * grad_weights
                        */

                        for (int j = 0; j < input_size; j++){
                            neuralnet->layers[layer_indx].weights[i * input_size + j]-=learning_rate*neuralnet->layers[layer_indx].grad_weights[i * input_size + j];
                        }
                    }
                }
            }
        printf("Epoch %d, Loss: %f\n", ep, loss_avg);
        }
    }


    free(loss);
    free(grad_buffer);
    free(indices);

    return 0;
}
int main(){
	NeuralNetwork* neuralnet = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	srand(time(NULL));
	NN_init(neuralnet,4);
	int input_size = 1;
	int n_examples = 100;
	int n_epochs = 100;
	int batch_size = 10;
	float learning_rate = 0.01;
	int out_size = 1;
	float *X = (float*)malloc(input_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) X[i]=drand48();
	float *Y = (float*)malloc(out_size*n_examples*sizeof(float));
	for (int i = 0; i < n_examples; i++) Y[i]=2*X[i]*X[i]+1;
	printf("Decida la funcion de activacion: 1)ReLU 2)Sigmoide 3)Tanh\n");
	int actfunction;
	scanf("%d",&actfunction);
	NN_Layer_innit(neuralnet,0,input_size,8,actfunction);
	NN_Layer_innit(neuralnet,1,8,8,actfunction);
	NN_Layer_innit(neuralnet,2,8,8,actfunction);
	NN_Layer_innit(neuralnet,3,8,out_size,actfunction);
	//NN_print_layer(neuralnet,0);
	//NN_print_layer(neuralnet,1);
	double start = omp_get_wtime();
	NN_loop(neuralnet,n_epochs,X,Y,n_examples,actfunction, batch_size, learning_rate);
	double end = omp_get_wtime();
	printf("Esto tardo %lf\n",end-start);
	return 0;
}

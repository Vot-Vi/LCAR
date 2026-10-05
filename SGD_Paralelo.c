#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    float *bias;    // i, hay uno por neurona
    float *input;
    float *output;
// BACKWARD
    float *grad_weights; // i,j donde i es la neurona y j es el input
    float *grad_bias;    // i, hay uno por neurona
    int initialized;
    DenseCache cache;
} DenseLayer;

typedef struct {
    int num_layers;
    DenseLayer *layers;
} NeuralNetwork;

/* Buffers privados de cada hilo (uno por capa) */
typedef struct {
    float **a;    // activación de la capa l
    float **z;    // pre-activación de la capa l
    float **dz;   // dL/dz de la capa l
    float **da;   // dL/da de la capa l
    float **gW;   // gradiente local de pesos
    float **gb;   // gradiente local de bias
} ThreadBuf;

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
    return 1.0f/(1.0f+expf(-x));
}

float Sigmoide_derivative(float x){
    return 1/(expf(x) + expf(-x) + 2);
}
//Fin Sigmoide

//Tanh
float Tanh(float x){
    return tanhf(x);
}

float Tanh_derivative(float x){
    return 4/(expf(x) + expf(-x) + 2);
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
    if (layer_indx>neuralnet->num_layers-1){
        printf("El indice de layer  %d es mayor que la capacidad inicializada de la nn\n",layer_indx);
        return -1;
    }
    if (layer_indx !=0 && input_size != neuralnet->layers[layer_indx-1].output_size){
        printf("El tamaño del output anterior no es igual a el input de esta capa \n");
        return -1;
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
    Layer->grad_weights = (float*)malloc((size_t)input_size*output_size*sizeof(float));
    Layer->grad_bias    = (float*)malloc(output_size*sizeof(float));
    Layer->cache.grad = (float*)malloc(output_size*sizeof(float));
    Layer->cache.z = (float*)malloc(output_size*sizeof(float));
    Layer->weights = (float*)malloc((size_t)input_size*output_size*sizeof(float));

    float ab = 0;
    if (actfunction == 1){
        ab = sqrtf(6/(float)input_size);
    }
    if (actfunction == 2 || actfunction == 3){
        ab = sqrtf(6/(float)(input_size + output_size));
    }

    for (int i = 0; i < output_size; i++){
        for (int j = 0; j < input_size; j++){
            float rnd = (float)drand48();
            Layer->weights[i*input_size+j] = 2*ab*rnd - ab; //random entre (-ab,ab)
        }
        Layer->bias[i] = 0.0f;
    }

    neuralnet->layers[layer_indx].initialized=1;
    return 0;
}

void NN_print_layer(NeuralNetwork* neuralnet,int layer_indx){
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

/* ---------- buffers por hilo ---------- */
ThreadBuf *alloc_thread_bufs(NeuralNetwork *nn, int nthreads)
{
    int L = nn->num_layers;
    ThreadBuf *tb = (ThreadBuf *)malloc(nthreads * sizeof(ThreadBuf));

    for (int t = 0; t < nthreads; t++) {
        tb[t].a  = (float **)malloc(L * sizeof(float *));
        tb[t].z  = (float **)malloc(L * sizeof(float *));
        tb[t].dz = (float **)malloc(L * sizeof(float *));
        tb[t].da = (float **)malloc(L * sizeof(float *));
        tb[t].gW = (float **)malloc(L * sizeof(float *));
        tb[t].gb = (float **)malloc(L * sizeof(float *));

        for (int l = 0; l < L; l++) {
            int in  = nn->layers[l].input_size;
            int out = nn->layers[l].output_size;
            tb[t].a[l]  = (float *)calloc(out, sizeof(float));
            tb[t].z[l]  = (float *)calloc(out, sizeof(float));
            tb[t].dz[l] = (float *)calloc(out, sizeof(float));
            tb[t].da[l] = (float *)calloc(out, sizeof(float));
            tb[t].gW[l] = (float *)calloc((size_t)in * out, sizeof(float));
            tb[t].gb[l] = (float *)calloc(out, sizeof(float));
        }
    }
    return tb;
}

void free_thread_bufs(ThreadBuf *tb, int nthreads, int L)
{
    for (int t = 0; t < nthreads; t++) {
        for (int l = 0; l < L; l++) {
            free(tb[t].a[l]);  free(tb[t].z[l]);  free(tb[t].dz[l]);
            free(tb[t].da[l]); free(tb[t].gW[l]); free(tb[t].gb[l]);
        }
        free(tb[t].a);  free(tb[t].z);  free(tb[t].dz);
        free(tb[t].da); free(tb[t].gW); free(tb[t].gb);
    }
    free(tb);
}

int NN_loop(NeuralNetwork *neuralnet, int epochs, float *X, float *Y, int n_examples, int actfunction, int batch_size, float learning_rate){

    int num_layers = neuralnet->num_layers;
    int final_size = neuralnet->layers[num_layers - 1].output_size;

    /* ?~Mndices para hacer shuffle */
    int *indices = (int *)malloc(n_examples * sizeof(int));

    /* Activación elegida con punteros a función (un solo bucle de entrenamiento) */
    float (*act)(float);
    float (*dact)(float);

    if (actfunction == 1)      { act = RELU;     dact = RELU_derivative;     }
    else if (actfunction == 2) { act = Sigmoide; dact = Sigmoide_derivative; }
    else                       { act = Tanh;     dact = Tanh_derivative;     }

    /* Buffers privados por hilo, reservados una sola vez */
    int nthreads = omp_get_max_threads();
    ThreadBuf *tbufs = alloc_thread_bufs(neuralnet, nthreads);

    /*
        ============================
        INICIO DEL ENTRENAMIENTO
        ============================
    */
    for (int ep = 0; ep < epochs; ep++){
        /* Shuffle de los ejemplos */
        random_permutation(indices, n_examples);

        double loss_avg = 0.0;

        /* ===== RECORREMOS LOS MINIBATCHES ===== */
        for (int batch_start = 0; batch_start < n_examples; batch_start += batch_size){

            /* Tamaño real del batch (importante para el último minibatch) */
            int current_batch_size = batch_size;
            if (batch_start + batch_size > n_examples) {
                current_batch_size = n_examples - batch_start;
            }

            /* ===== PONER LOS GRADIENTES GLOBALES EN CERO ===== */
            for (int layer_indx = 0; layer_indx < num_layers; layer_indx++){
                int input_size  = neuralnet->layers[layer_indx].input_size;
                int output_size = neuralnet->layers[layer_indx].output_size;

                for (int i = 0; i < output_size; i++) {
                    neuralnet->layers[layer_indx].grad_bias[i] = 0.0f;
                }
                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < input_size; j++) {
                        neuralnet->layers[layer_indx].grad_weights[i * input_size + j] = 0.0f;
                    }
                }
            }

            /* ===== EJEMPLOS DEL BATCH EN PARALELO ===== */
            #pragma omp parallel num_threads(nthreads) reduction(+:loss_avg)
            {
                ThreadBuf *tb = &tbufs[omp_get_thread_num()];

                /* gradientes locales en cero */
                for (int l = 0; l < num_layers; l++) {
                    int in  = neuralnet->layers[l].input_size;
                    int out = neuralnet->layers[l].output_size;
                    memset(tb->gW[l], 0, (size_t)in * out * sizeof(float));
                    memset(tb->gb[l], 0, out * sizeof(float));
                }

                #pragma omp for schedule(static)
                for (int b = 0; b < current_batch_size; b++) {
                    int ex = indices[batch_start + b];
                    const float *x = X + (size_t)ex * neuralnet->layers[0].input_size;
                    const float *y = Y + (size_t)ex * final_size;

                    /* ---------- FORWARD ---------- */
                    for (int l = 0; l < num_layers; l++) {
                        DenseLayer *L = &neuralnet->layers[l];
                        const float *in_vec = (l == 0) ? x : tb->a[l - 1];

                        for (int i = 0; i < L->output_size; i++) {
                            float acc = 0.0f;
                            const float *w = &L->weights[(size_t)i * L->input_size];
                            for (int j = 0; j < L->input_size; j++)
                                acc += w[j] * in_vec[j];
                            acc += L->bias[i];

                            tb->z[l][i] = acc;
                            /* la última capa es lineal (regresión) */
                            tb->a[l][i] = (l == num_layers - 1) ? acc : act(acc);
                        }
                    }

                    /* ---------- LOSS + dL/da de la última capa ---------- */
                    int last = num_layers - 1;
                    for (int i = 0; i < final_size; i++) {
                        float error = tb->a[last][i] - y[i];
                        tb->da[last][i] = error;
                        loss_avg += (0.5 * error * error / final_size) / n_examples;
                    }

                    /* ---------- BACKWARD ---------- */
                    for (int l = num_layers - 1; l >= 0; l--) {
                        DenseLayer *L = &neuralnet->layers[l];
                        const float *in_vec = (l == 0) ? x : tb->a[l - 1];

                        /* dz = da * act'(z) y acumulación de gradientes locales */
                        for (int i = 0; i < L->output_size; i++) {
                            float d = (l == num_layers - 1) ? 1.0f : dact(tb->z[l][i]);
                            float dz = tb->da[l][i] * d;
                            tb->dz[l][i] = dz;

                            tb->gb[l][i] += dz;
                            float *gw = &tb->gW[l][(size_t)i * L->input_size];
                            for (int j = 0; j < L->input_size; j++)
                                gw[j] += dz * in_vec[j];
                        }

                        /* da de la capa anterior = W^T * dz */
                        if (l > 0) {
                            for (int j = 0; j < L->input_size; j++) {
                                float sum = 0.0f;
                                for (int i = 0; i < L->output_size; i++)
                                    sum += tb->dz[l][i] * L->weights[(size_t)i * L->input_size + j];
                                tb->da[l - 1][j] = sum;
                            }
                        }
                    }
                } /* fin omp for (barrera implícita) */

                /* combinar gradientes locales en los globales */
                #pragma omp critical
                {
                    for (int l = 0; l < num_layers; l++) {
                        DenseLayer *L = &neuralnet->layers[l];
                        int n = L->input_size * L->output_size;
                        for (int k = 0; k < n; k++)
                            L->grad_weights[k] += tb->gW[l][k];
                        for (int i = 0; i < L->output_size; i++)
                            L->grad_bias[i] += tb->gb[l][i];
                    }
                }
            } /* fin parallel */

            /* ===== PROMEDIAR LOS GRADIENTES DEL MINIBATCH ===== */
            for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
            {
                int input_size  = neuralnet->layers[layer_indx].input_size;
                int output_size = neuralnet->layers[layer_indx].output_size;

                for (int i = 0; i < output_size; i++)
                {
                    neuralnet->layers[layer_indx].grad_bias[i] /= current_batch_size;
                }

                for (int i = 0; i < output_size; i++)
                {
                    for (int j = 0; j < input_size; j++){
                        neuralnet->layers[layer_indx].grad_weights[i * input_size + j] /= current_batch_size;
                    }
                }
            }

            /* ===== ACTUALIZAR PESOS ===== */
            for (int layer_indx = 0; layer_indx < num_layers; layer_indx++)
            {
                int input_size  = neuralnet->layers[layer_indx].input_size;
                int output_size = neuralnet->layers[layer_indx].output_size;

                for (int i = 0; i < output_size; i++)
                {
                    neuralnet->layers[layer_indx].bias[i] -= learning_rate * neuralnet->layers[layer_indx].grad_bias[i];

                    for (int j = 0; j < input_size; j++){
                        neuralnet->layers[layer_indx].weights[i * input_size + j] -= learning_rate * neuralnet->layers[layer_indx].grad_weights[i * input_size + j];
                    }
                }
            }
        }
        printf("Epoch %d, Loss: %f\n", ep, loss_avg);
    }

    free_thread_bufs(tbufs, nthreads, num_layers);
    free(indices);

    return 0;
}

int main(){
    NeuralNetwork* neuralnet = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    srand(420);
    srand48(420);
    NN_init(neuralnet,4);
    int input_size = 1;
    int n_examples = pow(2,20);
    int n_epochs = 100;
    int batch_size = 512;
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

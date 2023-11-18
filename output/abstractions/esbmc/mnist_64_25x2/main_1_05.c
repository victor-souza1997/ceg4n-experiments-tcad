

#include <stdlib.h>
#include <stdio.h>

#include "original.h"
#include "quantized.h"

#ifndef EQUIVALENCE
#define EQUIVALENCE 0
#endif

#ifndef EPSILON
#define EPSILON -1
#endif

#define BATCH 1

#ifndef WIDTH
#define WIDTH 64
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE 10
#endif

float nondet_float();

const float lower_bounds[BATCH][WIDTH] = 
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00098039209842681884765625, 0.49117648601531982421875, 0.455882370471954345703125, 0.00098039209842681884765625, 0, 0, 0, 0, 0.24019609391689300537109375, 0.769607841968536376953125, 0.6088235378265380859375, 0.3264705836772918701171875, 0, 0, 0, 0, 0.4127450883388519287109375, 0.3656862676143646240234375, 0.1068627536296844482421875, 0.518627464771270751953125, 0.0519607849419116973876953125, 0, 0, 0, 0.455882370471954345703125, 0.1892156898975372314453125, 0, 0.4049019515514373779296875, 0.1656862795352935791015625, 0, 0, 0, 0.3225490152835845947265625, 0.3343137204647064208984375, 0.06372548639774322509765625, 0.518627464771270751953125, 0.1147058904170989990234375, 0, 0, 0, 0.06372548639774322509765625, 0.495098054409027099609375, 0.530392169952392578125, 0.3382352888584136962890625, 0, 0, 0, 0, 0, 0, 0.0127451010048389434814453125, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.081372551620006561279296875, 0.073529414832592010498046875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.100980393588542938232421875, 0.591176509857177734375, 0.555882394313812255859375, 0.100980393588542938232421875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.3401961028575897216796875, 0.869607865810394287109375, 0.70882356166839599609375, 0.4264706075191497802734375, 0.0578431375324726104736328125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.512745082378387451171875, 0.4656862914562225341796875, 0.206862747669219970703125, 0.618627488613128662109375, 0.151960790157318115234375, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0617647059261798858642578125, 0.555882394313812255859375, 0.28921568393707275390625, 0.0539215691387653350830078125, 0.504901945590972900390625, 0.2656862735748291015625, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.4225490391254425048828125, 0.4343137443065643310546875, 0.16372549533843994140625, 0.618627488613128662109375, 0.214705884456634521484375, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.16372549533843994140625, 0.595098078250885009765625, 0.63039219379425048828125, 0.4382353127002716064453125, 0.073529414832592010498046875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.093137256801128387451171875, 0.112745106220245361328125, 0.0617647059261798858642578125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

static inline void init_symbolic_input(float input[BATCH * WIDTH])
{
    for(size_t i = 0; i < BATCH * WIDTH; i++)
    {
        input[i] = nondet_float();            
    }
}

static inline void add_input_assumptions(float input[BATCH * WIDTH], float x[BATCH][WIDTH])
{
    size_t i = 0;
    for(size_t b = 0; b < BATCH; b++)
    {
	    for(size_t w = 0; w < WIDTH; w++)
        {
            x[b][w] = input[i];
		    __ESBMC_assume(lower_bounds[b][w] <= x[b][w] && x[b][w] <= upper_bounds[b][w]);   
            i++;
        }
    }
}

static inline int top1(const float output[BATCH][OUTPUT_SIZE])
{
    int top = 0;
    for(size_t b = 0; b < BATCH; b++)
    {
        for (size_t o = 0; o < OUTPUT_SIZE; o++)
        {
            if(output[b][o] <= output[b][top])
            {
                continue;
            }
            top = (int) o;
        }
    }
    return top;
}

static inline void epsilon(
    const float output_original[BATCH][OUTPUT_SIZE],
    const float output_quantized[BATCH][OUTPUT_SIZE],
    float output_diff[BATCH][OUTPUT_SIZE]
){
    for(size_t b = 0; b < BATCH; b++)
    {
        for (size_t O = 0; O < OUTPUT_SIZE; O++)
        {
            output_diff[b][O] = output_original[b][O] - output_quantized[b][O];
            if(output_diff[b][O] >= 0)
            {
                continue;
            }
            output_diff[b][O] *= -1.0; 
        }
    }
}

static inline void check_top(const float output_original[BATCH][OUTPUT_SIZE], const float output_quantized[BATCH][OUTPUT_SIZE])
{
    int original_prediction = top1(output_original);
	int quantized_prediction = top1(output_quantized);

    int property_holds = (original_prediction==quantized_prediction);
    __ESBMC_assert(property_holds, "Networks not equivalent.");
}

static inline void check_epsilon(const float output_original[BATCH][OUTPUT_SIZE], const float output_quantized[BATCH][OUTPUT_SIZE])
{
    float output_diff[BATCH][OUTPUT_SIZE];
    
    // Get output diff
    epsilon(output_original, output_quantized, output_diff);

    int property_holds = 1;
    for(size_t O = 0; O < OUTPUT_SIZE; O++)
    {
        if (output_diff[0][O] <= EPSILON)
        {
            continue;
        }

        property_holds = 0;
        break;
    }

    __ESBMC_assert(property_holds, "Networks not equivalent.");
}

int main()
{

    // Define input and output vectors
    float input[BATCH*WIDTH];
    float x[BATCH][WIDTH];
    float original_output[BATCH][OUTPUT_SIZE];
    float quantized_output[BATCH][OUTPUT_SIZE];


    // Init symbolic input vector
    init_symbolic_input(input);

    // Add input assumptions
    add_input_assumptions(input, x);

    // Call networks
    original(x, original_output);
    quantized(x, quantized_output);

    if(EQUIVALENCE == 0)
    {
        check_top(original_output, quantized_output);
    } else {
        check_epsilon(original_output, quantized_output);
    }
}




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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1578431427478790283203125, 0.4284313619136810302734375, 0.3499999940395355224609375, 0.00098039209842681884765625, 0, 0, 0, 0.3264705836772918701171875, 0.628431379795074462890625, 0.3147058784961700439453125, 0.565686285495758056640625, 0.1617647111415863037109375, 0, 0, 0.3343137204647064208984375, 0.597058832645416259765625, 0.1264705955982208251953125, 0.0598039217293262481689453125, 0.53823530673980712890625, 0.0833333432674407958984375, 0, 0.20490197837352752685546875, 0.56960785388946533203125, 0.1029411852359771728515625, 0.1068627536296844482421875, 0.471568644046783447265625, 0.2872548997402191162109375, 0, 0, 0.467647075653076171875, 0.50686275959014892578125, 0.4049019515514373779296875, 0.487254917621612548828125, 0.1735294163227081298828125, 0, 0, 0, 0.1852941215038299560546875, 0.463725507259368896484375, 0.2833333313465118408203125, 0.0245098061859607696533203125, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0617647059261798858642578125, 0.25784313678741455078125, 0.528431355953216552734375, 0.4500000178813934326171875, 0.100980393588542938232421875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.073529414832592010498046875, 0.4264706075191497802734375, 0.728431403636932373046875, 0.4147059023380279541015625, 0.665686309337615966796875, 0.261764705181121826171875, 0.0500000007450580596923828125, 0.073529414832592010498046875, 0.4343137443065643310546875, 0.697058856487274169921875, 0.22647058963775634765625, 0.159803926944732666015625, 0.6382353305816650390625, 0.183333337306976318359375, 0.0500000007450580596923828125, 0.3049019873142242431640625, 0.6696078777313232421875, 0.2029411792755126953125, 0.206862747669219970703125, 0.571568667888641357421875, 0.3872549235820770263671875, 0.0578431375324726104736328125, 0.0500000007450580596923828125, 0.56764709949493408203125, 0.6068627834320068359375, 0.504901945590972900390625, 0.587254941463470458984375, 0.27352941036224365234375, 0.065686278045177459716796875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.285294115543365478515625, 0.563725531101226806640625, 0.3833333551883697509765625, 0.1245098114013671875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

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


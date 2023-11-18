

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1343137323856353759765625, 0.3343137204647064208984375, 0.0166666693985462188720703125, 0, 0, 0, 0, 0.1696078479290008544921875, 0.675490200519561767578125, 0.793137252330780029296875, 0.0990196168422698974609375, 0, 0, 0, 0.0794117748737335205078125, 0.52254903316497802734375, 0.3460784256458282470703125, 0.60098040103912353515625, 0, 0, 0, 0, 0.4166666567325592041015625, 0.23235295712947845458984375, 0.1696078479290008544921875, 0.557843148708343505859375, 0, 0, 0, 0.0950980484485626220703125, 0.5147058963775634765625, 0.0480392165482044219970703125, 0.4205882251262664794921875, 0.2833333313465118408203125, 0, 0, 0, 0.07156862318515777587890625, 0.52254903316497802734375, 0.467647075653076171875, 0.2872548997402191162109375, 0, 0, 0, 0, 0, 0.0911764800548553466796875, 0.0833333432674407958984375, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.2343137264251708984375, 0.4343137443065643310546875, 0.11666667461395263671875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.269607841968536376953125, 0.775490224361419677734375, 0.893137276172637939453125, 0.199019610881805419921875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.17941176891326904296875, 0.6225490570068359375, 0.4460784494876861572265625, 0.7009804248809814453125, 0.081372551620006561279296875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.069607846438884735107421875, 0.5166666507720947265625, 0.3323529660701751708984375, 0.269607841968536376953125, 0.657843172550201416015625, 0.069607846438884735107421875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.19509804248809814453125, 0.61470592021942138671875, 0.14803922176361083984375, 0.520588219165802001953125, 0.3833333551883697509765625, 0.0539215691387653350830078125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.1715686321258544921875, 0.6225490570068359375, 0.56764709949493408203125, 0.3872549235820770263671875, 0.081372551620006561279296875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.191176474094390869140625, 0.183333337306976318359375, 0.0578431375324726104736328125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

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


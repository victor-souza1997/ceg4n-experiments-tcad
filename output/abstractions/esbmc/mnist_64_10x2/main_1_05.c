

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3068627417087554931640625, 0.3225490152835845947265625, 0, 0, 0, 0, 0, 0.20882354676723480224609375, 0.3029411733150482177734375, 0.4166666567325592041015625, 0.3343137204647064208984375, 0, 0, 0, 0.0480392165482044219970703125, 0.3225490152835845947265625, 0, 0, 0.2794117629528045654296875, 0.2598039209842681884765625, 0, 0, 0.1852941215038299560546875, 0.1107843220233917236328125, 0, 0, 0.23627452552318572998046875, 0.2990196049213409423828125, 0, 0, 0.22843138873577117919921875, 0.0480392165482044219970703125, 0.00882352888584136962890625, 0.2637254893779754638671875, 0.3970588147640228271484375, 0.0284313745796680450439453125, 0, 0, 0.1617647111415863037109375, 0.3774509727954864501953125, 0.4284313619136810302734375, 0.2872548997402191162109375, 0.00882352888584136962890625, 0, 0, 0, 0, 0.00882352888584136962890625, 0.00490196049213409423828125, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0617647059261798858642578125, 0.0617647059261798858642578125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.077450983226299285888671875, 0.4068627655506134033203125, 0.4225490391254425048828125, 0.073529414832592010498046875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.3088235557079315185546875, 0.4029411971569061279296875, 0.5166666507720947265625, 0.4343137443065643310546875, 0.097058825194835662841796875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.14803922176361083984375, 0.4225490391254425048828125, 0.085294120013713836669921875, 0.085294120013713836669921875, 0.3794117867946624755859375, 0.3598039448261260986328125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.285294115543365478515625, 0.21078431606292724609375, 0.0500000007450580596923828125, 0.0578431375324726104736328125, 0.3362745344638824462890625, 0.3990196287631988525390625, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.3284313976764678955078125, 0.14803922176361083984375, 0.108823530375957489013671875, 0.3637255132198333740234375, 0.4970588386058807373046875, 0.128431379795074462890625, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.261764705181121826171875, 0.4774509966373443603515625, 0.528431355953216552734375, 0.3872549235820770263671875, 0.108823530375957489013671875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0617647059261798858642578125, 0.108823530375957489013671875, 0.104901961982250213623046875, 0.0578431375324726104736328125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

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


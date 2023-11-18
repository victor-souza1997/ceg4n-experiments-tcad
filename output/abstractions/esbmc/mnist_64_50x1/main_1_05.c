

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
{{0, 0, 0, 0, 0.0166666693985462188720703125, 0, 0, 0, 0, 0, 0.1264705955982208251953125, 0.55392158031463623046875, 0.3970588147640228271484375, 0.3970588147640228271484375, 0.06764705479145050048828125, 0, 0, 0.00882352888584136962890625, 0.557843148708343505859375, 0.3264705836772918701171875, 0.0127451010048389434814453125, 0.3068627417087554931640625, 0.471568644046783447265625, 0, 0, 0.2754901945590972900390625, 0.550000011920928955078125, 0, 0, 0.0401960797607898712158203125, 0.56176471710205078125, 0.0558823533356189727783203125, 0, 0.471568644046783447265625, 0.2598039209842681884765625, 0, 0, 0.2794117629528045654296875, 0.49117648601531982421875, 0, 0, 0.4441176354885101318359375, 0.4245097935199737548828125, 0.1264705955982208251953125, 0.3892156779766082763671875, 0.60098040103912353515625, 0.1303921639919281005859375, 0, 0, 0.0911764800548553466796875, 0.534313738346099853515625, 0.628431379795074462890625, 0.3892156779766082763671875, 0.0480392165482044219970703125, 0, 0, 0, 0, 0, 0.0127451010048389434814453125, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.089215688407421112060546875, 0.11666667461395263671875, 0.077450983226299285888671875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.22647058963775634765625, 0.653921604156494140625, 0.4970588386058807373046875, 0.4970588386058807373046875, 0.167647063732147216796875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.108823530375957489013671875, 0.657843172550201416015625, 0.4264706075191497802734375, 0.112745106220245361328125, 0.4068627655506134033203125, 0.571568667888641357421875, 0.073529414832592010498046875, 0.0500000007450580596923828125, 0.3754902184009552001953125, 0.650000035762786865234375, 0.093137256801128387451171875, 0.0500000007450580596923828125, 0.1401960849761962890625, 0.66176474094390869140625, 0.155882358551025390625, 0.0500000007450580596923828125, 0.571568667888641357421875, 0.3598039448261260986328125, 0.0500000007450580596923828125, 0.065686278045177459716796875, 0.3794117867946624755859375, 0.591176509857177734375, 0.097058825194835662841796875, 0.0500000007450580596923828125, 0.544117629528045654296875, 0.52450978755950927734375, 0.22647058963775634765625, 0.4892157018184661865234375, 0.7009804248809814453125, 0.230392158031463623046875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.191176474094390869140625, 0.634313762187957763671875, 0.728431403636932373046875, 0.4892157018184661865234375, 0.14803922176361083984375, 0.0539215691387653350830078125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.085294120013713836669921875, 0.112745106220245361328125, 0.0617647059261798858642578125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

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


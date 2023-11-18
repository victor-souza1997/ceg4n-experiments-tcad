

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1107843220233917236328125, 0.4441176354885101318359375, 0.53823530673980712890625, 0.3696078360080718994140625, 0.0441176481544971466064453125, 0, 0, 0.0441176481544971466064453125, 0.52254903316497802734375, 0.22843138873577117919921875, 0.1696078479290008544921875, 0.53823530673980712890625, 0.557843148708343505859375, 0.00490196049213409423828125, 0, 0.3539215624332427978515625, 0.4833333492279052734375, 0, 0, 0.1500000059604644775390625, 0.761764705181121826171875, 0.1421568691730499267578125, 0, 0.49117648601531982421875, 0.3382352888584136962890625, 0, 0, 0.24411766231060028076171875, 0.644117653369903564453125, 0.0362745113670825958251953125, 0, 0.3303921520709991455078125, 0.61666667461395263671875, 0.22058825194835662841796875, 0.2872548997402191162109375, 0.6401960849761962890625, 0.2637254893779754638671875, 0, 0, 0.00490196049213409423828125, 0.3578431308269500732421875, 0.56176471710205078125, 0.530392169952392578125, 0.24803923070430755615234375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.065686278045177459716796875, 0.077450983226299285888671875, 0.0539215691387653350830078125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.21078431606292724609375, 0.544117629528045654296875, 0.6382353305816650390625, 0.4696078598499298095703125, 0.144117653369903564453125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.144117653369903564453125, 0.6225490570068359375, 0.3284313976764678955078125, 0.269607841968536376953125, 0.6382353305816650390625, 0.657843172550201416015625, 0.104901961982250213623046875, 0.0500000007450580596923828125, 0.4539215862751007080078125, 0.58333337306976318359375, 0.065686278045177459716796875, 0.0500000007450580596923828125, 0.25, 0.861764729022979736328125, 0.24215686321258544921875, 0.0500000007450580596923828125, 0.591176509857177734375, 0.4382353127002716064453125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.3441176712512969970703125, 0.744117677211761474609375, 0.136274516582489013671875, 0.0500000007450580596923828125, 0.4303921759128570556640625, 0.716666698455810546875, 0.3205882608890533447265625, 0.3872549235820770263671875, 0.74019610881805419921875, 0.3637255132198333740234375, 0.0539215691387653350830078125, 0.0500000007450580596923828125, 0.104901961982250213623046875, 0.4578431546688079833984375, 0.66176474094390869140625, 0.63039219379425048828125, 0.3480392396450042724609375, 0.065686278045177459716796875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0539215691387653350830078125, 0.081372551620006561279296875, 0.069607846438884735107421875, 0.0500000007450580596923828125, 0.0500000007450580596923828125, 0.0500000007450580596923828125}};

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


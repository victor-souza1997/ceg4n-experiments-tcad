

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.13078431785106658935546875, 0.46411764621734619140625, 0.558235347270965576171875, 0.389607846736907958984375, 0.064117647707462310791015625, 0, 0, 0.064117647707462310791015625, 0.542549073696136474609375, 0.248431384563446044921875, 0.18960784375667572021484375, 0.558235347270965576171875, 0.577843189239501953125, 0.02490196190774440765380859375, 0, 0.373921573162078857421875, 0.503333389759063720703125, 0, 0, 0.17000000178813934326171875, 0.7817647457122802734375, 0.16215686500072479248046875, 0, 0.511176526546478271484375, 0.358235299587249755859375, 0, 0, 0.264117658138275146484375, 0.66411769390106201171875, 0.056274510920047760009765625, 0, 0.350392162799835205078125, 0.636666715145111083984375, 0.240588247776031494140625, 0.30725491046905517578125, 0.660196125507354736328125, 0.2837255001068115234375, 0, 0, 0.02490196190774440765380859375, 0.3778431415557861328125, 0.581764757633209228515625, 0.550392210483551025390625, 0.268039226531982421875, 0, 0, 0, 0, 0, 0.00137255154550075531005859375, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0456862747669219970703125, 0.0574509799480438232421875, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.19078432023525238037109375, 0.524117648601531982421875, 0.618235290050506591796875, 0.44960784912109375, 0.124117650091648101806640625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.124117650091648101806640625, 0.602549016475677490234375, 0.3084313869476318359375, 0.24960784614086151123046875, 0.618235290050506591796875, 0.63784313201904296875, 0.0849019587039947509765625, 0.02999999932944774627685546875, 0.4339215755462646484375, 0.563333332538604736328125, 0.0456862747669219970703125, 0.02999999932944774627685546875, 0.23000000417232513427734375, 0.8417646884918212890625, 0.22215686738491058349609375, 0.02999999932944774627685546875, 0.571176469326019287109375, 0.418235301971435546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.3241176605224609375, 0.72411763668060302734375, 0.116274513304233551025390625, 0.02999999932944774627685546875, 0.41039216518402099609375, 0.696666657924652099609375, 0.30058825016021728515625, 0.367254912853240966796875, 0.720196068286895751953125, 0.343725502490997314453125, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.0849019587039947509765625, 0.437843143939971923828125, 0.641764700412750244140625, 0.610392153263092041015625, 0.328039228916168212890625, 0.0456862747669219970703125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.0613725483417510986328125, 0.0496078431606292724609375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


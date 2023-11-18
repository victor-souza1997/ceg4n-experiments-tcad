

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
{{0, 0, 0, 0, 0.00529411993920803070068359375, 0.032745100557804107666015625, 0, 0, 0, 0, 0, 0.15823529660701751708984375, 0.558235347270965576171875, 0.644509851932525634765625, 0.224901974201202392578125, 0, 0, 0, 0.071960784494876861572265625, 0.589607894420623779296875, 0.4915686547756195068359375, 0.511176526546478271484375, 0.32294118404388427734375, 0, 0, 0, 0, 0.052352942526340484619140625, 0.12686274945735931396484375, 0.550392210483551025390625, 0.10333333909511566162109375, 0, 0, 0, 0.068039216101169586181640625, 0.256274521350860595703125, 0.652352988719940185546875, 0.366078436374664306640625, 0, 0, 0, 0.26019608974456787109375, 0.605294167995452880859375, 0.793529450893402099609375, 0.566078484058380126953125, 0.526862800121307373046875, 0.079803921282291412353515625, 0, 0.052352942526340484619140625, 0.52294123172760009765625, 0.46411764621734619140625, 0.19352941215038299560546875, 0, 0.22882354259490966796875, 0.19745098054409027099609375, 0, 0, 0.02490196190774440765380859375, 0.00529411993920803070068359375, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0652941167354583740234375, 0.092745102941989898681640625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0378431379795074462890625, 0.21823529899120330810546875, 0.618235290050506591796875, 0.704509794712066650390625, 0.28490197658538818359375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1319607794284820556640625, 0.649607837200164794921875, 0.55156862735748291015625, 0.571176469326019287109375, 0.382941186428070068359375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0535294115543365478515625, 0.112352944910526275634765625, 0.18686275184154510498046875, 0.610392153263092041015625, 0.16333334147930145263671875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.1280392110347747802734375, 0.31627452373504638671875, 0.712352931499481201171875, 0.42607843875885009765625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.0496078431606292724609375, 0.320196092128753662109375, 0.665294110774993896484375, 0.853529393672943115234375, 0.626078426837921142578125, 0.586862742900848388671875, 0.1398039162158966064453125, 0.02999999932944774627685546875, 0.112352944910526275634765625, 0.58294117450714111328125, 0.524117648601531982421875, 0.2535293996334075927734375, 0.0574509799480438232421875, 0.288823544979095458984375, 0.2574509680271148681640625, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.0849019587039947509765625, 0.0652941167354583740234375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0378431379795074462890625, 0.0417647063732147216796875, 0.02999999932944774627685546875}};

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


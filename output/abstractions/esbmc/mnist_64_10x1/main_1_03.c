

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17784313857555389404296875, 0.44843137264251708984375, 0.37000000476837158203125, 0.02098039351403713226318359375, 0, 0, 0, 0.3464705944061279296875, 0.64843142032623291015625, 0.334705889225006103515625, 0.58568632602691650390625, 0.18176470696926116943359375, 0, 0, 0.35431373119354248046875, 0.61705887317657470703125, 0.14647059142589569091796875, 0.079803921282291412353515625, 0.558235347270965576171875, 0.10333333909511566162109375, 0, 0.224901974201202392578125, 0.589607894420623779296875, 0.12294118106365203857421875, 0.12686274945735931396484375, 0.4915686547756195068359375, 0.30725491046905517578125, 0, 0, 0.4876470863819122314453125, 0.526862800121307373046875, 0.4249019622802734375, 0.50725495815277099609375, 0.19352941215038299560546875, 0, 0, 0, 0.20529411733150482177734375, 0.4837255179882049560546875, 0.303333342075347900390625, 0.044509805738925933837890625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0417647063732147216796875, 0.23784314095973968505859375, 0.508431375026702880859375, 0.430000007152557373046875, 0.0809803903102874755859375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0535294115543365478515625, 0.406470596790313720703125, 0.70843136310577392578125, 0.39470589160919189453125, 0.64568626880645751953125, 0.24176470935344696044921875, 0.02999999932944774627685546875, 0.0535294115543365478515625, 0.414313733577728271484375, 0.67705881595611572265625, 0.20647059381008148193359375, 0.1398039162158966064453125, 0.618235290050506591796875, 0.16333334147930145263671875, 0.02999999932944774627685546875, 0.28490197658538818359375, 0.649607837200164794921875, 0.18294118344783782958984375, 0.18686275184154510498046875, 0.55156862735748291015625, 0.367254912853240966796875, 0.0378431379795074462890625, 0.02999999932944774627685546875, 0.547647058963775634765625, 0.586862742900848388671875, 0.484901964664459228515625, 0.56725490093231201171875, 0.2535293996334075927734375, 0.0456862747669219970703125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.2652941048145294189453125, 0.543725490570068359375, 0.36333334445953369140625, 0.104509808123111724853515625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


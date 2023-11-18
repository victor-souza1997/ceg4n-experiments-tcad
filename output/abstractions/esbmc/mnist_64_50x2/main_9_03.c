

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02098039351403713226318359375, 0.09549020230770111083984375, 0.087647058069705963134765625, 0, 0, 0, 0, 0.268039226531982421875, 0.43274509906768798828125, 0.405294120311737060546875, 0.41705882549285888671875, 0.14254902303218841552734375, 0.19352941215038299560546875, 0, 0, 0.452352941036224365234375, 0.29156863689422607421875, 0.19745098054409027099609375, 0.38568627834320068359375, 0.37000000476837158203125, 0.26019608974456787109375, 0, 0, 0.17784313857555389404296875, 0.71117651462554931640625, 0.456274509429931640625, 0.10333333909511566162109375, 0, 0, 0, 0, 0.29156863689422607421875, 0.436666667461395263671875, 0.428823530673980712890625, 0, 0, 0, 0, 0, 0.232745110988616943359375, 0.452352941036224365234375, 0.43274509906768798828125, 0, 0, 0, 0, 0, 0.01705882512032985687255859375, 0.19745098054409027099609375, 0.064117647707462310791015625, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.0809803903102874755859375, 0.15549020469188690185546875, 0.1476470530033111572265625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0378431379795074462890625, 0.328039228916168212890625, 0.492745101451873779296875, 0.4652941226959228515625, 0.477058827877044677734375, 0.20254902541637420654296875, 0.2535293996334075927734375, 0.02999999932944774627685546875, 0.0417647063732147216796875, 0.51235294342041015625, 0.351568639278411865234375, 0.2574509680271148681640625, 0.445686280727386474609375, 0.430000007152557373046875, 0.320196092128753662109375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.23784314095973968505859375, 0.77117645740509033203125, 0.516274511814117431640625, 0.16333334147930145263671875, 0.0496078431606292724609375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.351568639278411865234375, 0.4966666698455810546875, 0.48882353305816650390625, 0.0456862747669219970703125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.292745113372802734375, 0.51235294342041015625, 0.492745101451873779296875, 0.0496078431606292724609375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0770588219165802001953125, 0.2574509680271148681640625, 0.124117650091648101806640625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


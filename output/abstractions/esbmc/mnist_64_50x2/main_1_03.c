

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15431372821331024169921875, 0.35431373119354248046875, 0.036666668951511383056640625, 0, 0, 0, 0, 0.18960784375667572021484375, 0.69549024105072021484375, 0.8131372928619384765625, 0.11901961266994476318359375, 0, 0, 0, 0.09941177070140838623046875, 0.542549073696136474609375, 0.366078436374664306640625, 0.620980441570281982421875, 0.00137255154550075531005859375, 0, 0, 0, 0.436666667461395263671875, 0.2523529529571533203125, 0.18960784375667572021484375, 0.577843189239501953125, 0, 0, 0, 0.11509804427623748779296875, 0.534705936908721923828125, 0.068039216101169586181640625, 0.4405882358551025390625, 0.303333342075347900390625, 0, 0, 0, 0.091568626463413238525390625, 0.542549073696136474609375, 0.4876470863819122314453125, 0.30725491046905517578125, 0.00137255154550075531005859375, 0, 0, 0, 0, 0.11117647588253021240234375, 0.10333333909511566162109375, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.21431373059749603271484375, 0.414313733577728271484375, 0.096666671335697174072265625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.24960784614086151123046875, 0.75549018383026123046875, 0.8731372356414794921875, 0.17901961505413055419921875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.15941177308559417724609375, 0.602549016475677490234375, 0.42607843875885009765625, 0.680980384349822998046875, 0.0613725483417510986328125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0496078431606292724609375, 0.4966666698455810546875, 0.312352955341339111328125, 0.24960784614086151123046875, 0.63784313201904296875, 0.0496078431606292724609375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.17509804666042327880859375, 0.594705879688262939453125, 0.1280392110347747802734375, 0.500588238239288330078125, 0.36333334445953369140625, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1515686213970184326171875, 0.602549016475677490234375, 0.547647058963775634765625, 0.367254912853240966796875, 0.0613725483417510986328125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.17117647826671600341796875, 0.16333334147930145263671875, 0.0378431379795074462890625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


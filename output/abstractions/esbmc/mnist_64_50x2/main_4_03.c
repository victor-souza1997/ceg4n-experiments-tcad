

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0.02490196190774440765380859375, 0.17392157018184661865234375, 0.22882354259490966796875, 0.24450981616973876953125, 0.091568626463413238525390625, 0, 0, 0, 0.075882352888584136962890625, 0.37000000476837158203125, 0.295490205287933349609375, 0.511176526546478271484375, 0.60137259960174560546875, 0.01705882512032985687255859375, 0, 0, 0, 0, 0.060196079313755035400390625, 0.620980441570281982421875, 0.636666715145111083984375, 0.319019615650177001953125, 0.01313725672662258148193359375, 0, 0, 0.040588237345218658447265625, 0.09941177070140838623046875, 0.060196079313755035400390625, 0.032745100557804107666015625, 0.326862752437591552734375, 0.303333342075347900390625, 0, 0, 0.040588237345218658447265625, 0.452352941036224365234375, 0.515098094940185546875, 0.50725495815277099609375, 0.668039262294769287109375, 0.37000000476837158203125, 0, 0, 0, 0.060196079313755035400390625, 0.248431384563446044921875, 0.264117658138275146484375, 0.14647059142589569091796875, 0.00529411993920803070068359375, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0849019587039947509765625, 0.23392157256603240966796875, 0.288823544979095458984375, 0.304509818553924560546875, 0.1515686213970184326171875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1358823478221893310546875, 0.430000007152557373046875, 0.355490207672119140625, 0.571176469326019287109375, 0.66137254238128662109375, 0.0770588219165802001953125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.120196081697940826416015625, 0.680980384349822998046875, 0.696666657924652099609375, 0.37901961803436279296875, 0.0731372535228729248046875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.100588239729404449462890625, 0.15941177308559417724609375, 0.120196081697940826416015625, 0.092745102941989898681640625, 0.38686275482177734375, 0.36333334445953369140625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.100588239729404449462890625, 0.51235294342041015625, 0.5750980377197265625, 0.56725490093231201171875, 0.728039205074310302734375, 0.430000007152557373046875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.120196081697940826416015625, 0.3084313869476318359375, 0.3241176605224609375, 0.20647059381008148193359375, 0.0652941167354583740234375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


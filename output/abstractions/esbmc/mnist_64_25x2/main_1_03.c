

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
{{0, 0, 0, 0.00137255154550075531005859375, 0, 0, 0, 0, 0, 0, 0.02098039351403713226318359375, 0.511176526546478271484375, 0.4758823812007904052734375, 0.02098039351403713226318359375, 0, 0, 0, 0, 0.26019608974456787109375, 0.78960788249969482421875, 0.628823578357696533203125, 0.3464705944061279296875, 0, 0, 0, 0, 0.43274509906768798828125, 0.38568627834320068359375, 0.12686274945735931396484375, 0.53862750530242919921875, 0.071960784494876861572265625, 0, 0, 0, 0.4758823812007904052734375, 0.20921568572521209716796875, 0, 0.4249019622802734375, 0.18568627536296844482421875, 0, 0, 0, 0.342549026012420654296875, 0.35431373119354248046875, 0.083725489675998687744140625, 0.53862750530242919921875, 0.13470588624477386474609375, 0, 0, 0, 0.083725489675998687744140625, 0.515098094940185546875, 0.550392210483551025390625, 0.358235299587249755859375, 0, 0, 0, 0, 0, 0.01313725672662258148193359375, 0.032745100557804107666015625, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0613725483417510986328125, 0.0535294115543365478515625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0809803903102874755859375, 0.571176469326019287109375, 0.53588235378265380859375, 0.0809803903102874755859375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.320196092128753662109375, 0.84960782527923583984375, 0.688823521137237548828125, 0.406470596790313720703125, 0.0378431379795074462890625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.492745101451873779296875, 0.445686280727386474609375, 0.18686275184154510498046875, 0.59862744808197021484375, 0.1319607794284820556640625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0417647063732147216796875, 0.53588235378265380859375, 0.2692156732082366943359375, 0.0339215695858001708984375, 0.484901964664459228515625, 0.24568627774715423583984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.4025490283966064453125, 0.414313733577728271484375, 0.1437254846096038818359375, 0.59862744808197021484375, 0.19470588862895965576171875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1437254846096038818359375, 0.5750980377197265625, 0.610392153263092041015625, 0.418235301971435546875, 0.0535294115543365478515625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0731372535228729248046875, 0.092745102941989898681640625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


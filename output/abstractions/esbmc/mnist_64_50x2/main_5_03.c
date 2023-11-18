

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.083725489675998687744140625, 0.10725490748882293701171875, 0, 0, 0, 0, 0.224901974201202392578125, 0.02490196190774440765380859375, 0.248431384563446044921875, 0.27588236331939697265625, 0, 0, 0, 0.00137255154550075531005859375, 0.511176526546478271484375, 0.224901974201202392578125, 0.519019663333892822265625, 0.27588236331939697265625, 0, 0, 0, 0, 0.271960794925689697265625, 0.4758823812007904052734375, 0.6719608306884765625, 0.14647059142589569091796875, 0, 0, 0, 0, 0, 0.15431372821331024169921875, 0.58568632602691650390625, 0.00921568833291530609130859375, 0, 0, 0, 0, 0, 0.087647058069705963134765625, 0.4837255179882049560546875, 0.01313725672662258148193359375, 0, 0, 0, 0, 0, 0, 0.068039216101169586181640625, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.1437254846096038818359375, 0.16725490987300872802734375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0456862747669219970703125, 0.28490197658538818359375, 0.0849019587039947509765625, 0.3084313869476318359375, 0.335882365703582763671875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0613725483417510986328125, 0.571176469326019287109375, 0.28490197658538818359375, 0.579019606113433837890625, 0.335882365703582763671875, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.33196079730987548828125, 0.53588235378265380859375, 0.731960773468017578125, 0.20647059381008148193359375, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.21431373059749603271484375, 0.64568626880645751953125, 0.0692156851291656494140625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1476470530033111572265625, 0.543725490570068359375, 0.0731372535228729248046875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.1280392110347747802734375, 0.0456862747669219970703125, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


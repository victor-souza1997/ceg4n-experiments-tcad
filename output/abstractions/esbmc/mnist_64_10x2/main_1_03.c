

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.326862752437591552734375, 0.342549026012420654296875, 0, 0, 0, 0, 0, 0.22882354259490966796875, 0.32294118404388427734375, 0.436666667461395263671875, 0.35431373119354248046875, 0.01705882512032985687255859375, 0, 0, 0.068039216101169586181640625, 0.342549026012420654296875, 0.00529411993920803070068359375, 0.00529411993920803070068359375, 0.299411773681640625, 0.279803931713104248046875, 0, 0, 0.20529411733150482177734375, 0.13078431785106658935546875, 0, 0, 0.256274521350860595703125, 0.319019615650177001953125, 0, 0, 0.248431384563446044921875, 0.068039216101169586181640625, 0.02882353030145168304443359375, 0.2837255001068115234375, 0.41705882549285888671875, 0.048431374132633209228515625, 0, 0, 0.18176470696926116943359375, 0.397450983524322509765625, 0.44843137264251708984375, 0.30725491046905517578125, 0.02882353030145168304443359375, 0, 0, 0, 0, 0.02882353030145168304443359375, 0.02490196190774440765380859375, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0417647063732147216796875, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0574509799480438232421875, 0.38686275482177734375, 0.4025490283966064453125, 0.0535294115543365478515625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0339215695858001708984375, 0.288823544979095458984375, 0.382941186428070068359375, 0.4966666698455810546875, 0.414313733577728271484375, 0.0770588219165802001953125, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.1280392110347747802734375, 0.4025490283966064453125, 0.0652941167354583740234375, 0.0652941167354583740234375, 0.359411776065826416015625, 0.3398039340972900390625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.2652941048145294189453125, 0.19078432023525238037109375, 0.02999999932944774627685546875, 0.0378431379795074462890625, 0.31627452373504638671875, 0.37901961803436279296875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.3084313869476318359375, 0.1280392110347747802734375, 0.0888235270977020263671875, 0.343725502490997314453125, 0.477058827877044677734375, 0.108431376516819000244140625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.24176470935344696044921875, 0.45745098590850830078125, 0.508431375026702880859375, 0.367254912853240966796875, 0.0888235270977020263671875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0417647063732147216796875, 0.0888235270977020263671875, 0.0849019587039947509765625, 0.0378431379795074462890625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


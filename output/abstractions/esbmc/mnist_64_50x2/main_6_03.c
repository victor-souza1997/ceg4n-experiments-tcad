

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02882353030145168304443359375, 0.071960784494876861572265625, 0.18568627536296844482421875, 0.35431373119354248046875, 0.17392157018184661865234375, 0, 0, 0, 0.43274509906768798828125, 0.50725495815277099609375, 0.4719608128070831298828125, 0.3464705944061279296875, 0.11901961266994476318359375, 0, 0, 0.02490196190774440765380859375, 0.4405882358551025390625, 0.12686274945735931396484375, 0.02882353030145168304443359375, 0, 0, 0, 0, 0.00921568833291530609130859375, 0.452352941036224365234375, 0.4954902231693267822265625, 0.405294120311737060546875, 0.21313725411891937255859375, 0, 0, 0, 0, 0.00529411993920803070068359375, 0.02882353030145168304443359375, 0.071960784494876861572265625, 0.4758823812007904052734375, 0.071960784494876861572265625, 0, 0, 0, 0.16215686500072479248046875, 0.5307843685150146484375, 0.589607894420623779296875, 0.50725495815277099609375, 0.02882353030145168304443359375, 0, 0, 0, 0.01313725672662258148193359375, 0.12294118106365203857421875, 0.13078431785106658935546875, 0.02098039351403713226318359375, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0888235270977020263671875, 0.1319607794284820556640625, 0.24568627774715423583984375, 0.414313733577728271484375, 0.23392157256603240966796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0456862747669219970703125, 0.492745101451873779296875, 0.56725490093231201171875, 0.531960785388946533203125, 0.406470596790313720703125, 0.17901961505413055419921875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0849019587039947509765625, 0.500588238239288330078125, 0.18686275184154510498046875, 0.0888235270977020263671875, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0692156851291656494140625, 0.51235294342041015625, 0.555490195751190185546875, 0.4652941226959228515625, 0.2731372416019439697265625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0652941167354583740234375, 0.0888235270977020263671875, 0.1319607794284820556640625, 0.53588235378265380859375, 0.1319607794284820556640625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.22215686738491058349609375, 0.5907843112945556640625, 0.649607837200164794921875, 0.56725490093231201171875, 0.0888235270977020263671875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0731372535228729248046875, 0.18294118344783782958984375, 0.19078432023525238037109375, 0.0809803903102874755859375, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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




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
{{0, 0, 0, 0.005686275660991668701171875, 0.017450980842113494873046875, 0, 0, 0, 0, 0, 0.150784313678741455078125, 0.4841176569461822509765625, 0.578235328197479248046875, 0.4096078574657440185546875, 0.0841176509857177734375, 0, 0, 0.0841176509857177734375, 0.562549054622650146484375, 0.2684313952922821044921875, 0.2096078395843505859375, 0.578235328197479248046875, 0.597843170166015625, 0.044901959598064422607421875, 0, 0.3939215838909149169921875, 0.523333370685577392578125, 0.005686275660991668701171875, 0, 0.189999997615814208984375, 0.8017647266387939453125, 0.182156860828399658203125, 0, 0.531176507472991943359375, 0.3782353103160858154296875, 0, 0, 0.2841176688671112060546875, 0.68411767482757568359375, 0.07627451419830322265625, 0, 0.3703921735286712646484375, 0.656666696071624755859375, 0.2605882585048675537109375, 0.3272549211978912353515625, 0.680196106433868408203125, 0.3037255108356475830078125, 0, 0, 0.044901959598064422607421875, 0.3978431522846221923828125, 0.601764738559722900390625, 0.570392191410064697265625, 0.2880392372608184814453125, 0.005686275660991668701171875, 0, 0, 0, 0, 0.02137255109846591949462890625, 0.009607844054698944091796875, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.0256862752139568328857421875, 0.0374509803950786590576171875, 0.01392156817018985748291015625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.1707843244075775146484375, 0.504117667675018310546875, 0.598235309123992919921875, 0.4296078383922576904296875, 0.10411764681339263916015625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.10411764681339263916015625, 0.582549035549163818359375, 0.2884313762187957763671875, 0.2296078503131866455078125, 0.598235309123992919921875, 0.617843151092529296875, 0.06490196287631988525390625, 0.00999999977648258209228515625, 0.4139215648174285888671875, 0.543333351612091064453125, 0.0256862752139568328857421875, 0.00999999977648258209228515625, 0.2100000083446502685546875, 0.8217647075653076171875, 0.2021568715572357177734375, 0.00999999977648258209228515625, 0.551176488399505615234375, 0.3982352912425994873046875, 0.00999999977648258209228515625, 0.01392156817018985748291015625, 0.3041176497936248779296875, 0.70411765575408935546875, 0.09627451002597808837890625, 0.00999999977648258209228515625, 0.3903921544551849365234375, 0.676666676998138427734375, 0.2805882394313812255859375, 0.3472549021244049072265625, 0.700196087360382080078125, 0.3237254917621612548828125, 0.01392156817018985748291015625, 0.00999999977648258209228515625, 0.06490196287631988525390625, 0.4178431332111358642578125, 0.621764719486236572265625, 0.590392172336578369140625, 0.3080392181873321533203125, 0.0256862752139568328857421875, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.01392156817018985748291015625, 0.04137255251407623291015625, 0.0296078436076641082763671875, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625}};

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


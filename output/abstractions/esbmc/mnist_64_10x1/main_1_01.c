

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
{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001764706335961818695068359375, 0.197843134403228759765625, 0.4684313833713531494140625, 0.3900000154972076416015625, 0.040980391204357147216796875, 0, 0, 0.013529412448406219482421875, 0.3664706051349639892578125, 0.66843140125274658203125, 0.3547058999538421630859375, 0.60568630695343017578125, 0.20176470279693603515625, 0, 0.013529412448406219482421875, 0.3743137419223785400390625, 0.63705885410308837890625, 0.166470587253570556640625, 0.099803924560546875, 0.578235328197479248046875, 0.123333342373371124267578125, 0, 0.24490197002887725830078125, 0.609607875347137451171875, 0.142941176891326904296875, 0.1468627452850341796875, 0.51156866550445556640625, 0.3272549211978912353515625, 0, 0, 0.507647097110748291015625, 0.546862781047821044921875, 0.4449019730091094970703125, 0.52725493907928466796875, 0.213529407978057861328125, 0.005686275660991668701171875, 0, 0, 0.2252941131591796875, 0.503725528717041015625, 0.3233333528041839599609375, 0.064509809017181396484375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.0217647068202495574951171875, 0.2178431451320648193359375, 0.4884313642978668212890625, 0.4099999964237213134765625, 0.06098039448261260986328125, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.0335294120013713836669921875, 0.3864705860614776611328125, 0.68843138217926025390625, 0.3747058808803558349609375, 0.62568628787994384765625, 0.2217647135257720947265625, 0.00999999977648258209228515625, 0.0335294120013713836669921875, 0.3943137228488922119140625, 0.65705883502960205078125, 0.1864705979824066162109375, 0.11980392038822174072265625, 0.598235309123992919921875, 0.1433333456516265869140625, 0.00999999977648258209228515625, 0.2649019658565521240234375, 0.629607856273651123046875, 0.1629411876201629638671875, 0.1668627560138702392578125, 0.53156864643096923828125, 0.3472549021244049072265625, 0.0178431384265422821044921875, 0.00999999977648258209228515625, 0.527647078037261962890625, 0.566862761974334716796875, 0.4649019539356231689453125, 0.54725492000579833984375, 0.2335294187068939208984375, 0.0256862752139568328857421875, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.2452941238880157470703125, 0.5237255096435546875, 0.3433333337306976318359375, 0.08450980484485626220703125, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625, 0.00999999977648258209228515625}};

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


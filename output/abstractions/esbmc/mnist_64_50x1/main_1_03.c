

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
{{0, 0, 0, 0.00921568833291530609130859375, 0.036666668951511383056640625, 0, 0, 0, 0, 0, 0.14647059142589569091796875, 0.573921620845794677734375, 0.41705882549285888671875, 0.41705882549285888671875, 0.087647058069705963134765625, 0, 0, 0.02882353030145168304443359375, 0.577843189239501953125, 0.3464705944061279296875, 0.032745100557804107666015625, 0.326862752437591552734375, 0.4915686547756195068359375, 0, 0, 0.295490205287933349609375, 0.57000005245208740234375, 0.01313725672662258148193359375, 0, 0.060196079313755035400390625, 0.581764757633209228515625, 0.075882352888584136962890625, 0, 0.4915686547756195068359375, 0.279803931713104248046875, 0, 0, 0.299411773681640625, 0.511176526546478271484375, 0.01705882512032985687255859375, 0, 0.46411764621734619140625, 0.444509804248809814453125, 0.14647059142589569091796875, 0.4092156887054443359375, 0.620980441570281982421875, 0.15039215981960296630859375, 0, 0, 0.11117647588253021240234375, 0.55431377887725830078125, 0.64843142032623291015625, 0.4092156887054443359375, 0.068039216101169586181640625, 0, 0, 0, 0, 0.00529411993920803070068359375, 0.032745100557804107666015625, 0, 0, 0, 0}};

const float upper_bounds[BATCH][WIDTH] = 
{{0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0692156851291656494140625, 0.096666671335697174072265625, 0.0574509799480438232421875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.20647059381008148193359375, 0.633921563625335693359375, 0.477058827877044677734375, 0.477058827877044677734375, 0.1476470530033111572265625, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0888235270977020263671875, 0.63784313201904296875, 0.406470596790313720703125, 0.092745102941989898681640625, 0.38686275482177734375, 0.55156862735748291015625, 0.0535294115543365478515625, 0.02999999932944774627685546875, 0.355490207672119140625, 0.62999999523162841796875, 0.0731372535228729248046875, 0.02999999932944774627685546875, 0.120196081697940826416015625, 0.641764700412750244140625, 0.1358823478221893310546875, 0.02999999932944774627685546875, 0.55156862735748291015625, 0.3398039340972900390625, 0.02999999932944774627685546875, 0.0456862747669219970703125, 0.359411776065826416015625, 0.571176469326019287109375, 0.0770588219165802001953125, 0.02999999932944774627685546875, 0.524117648601531982421875, 0.50450980663299560546875, 0.20647059381008148193359375, 0.469215691089630126953125, 0.680980384349822998046875, 0.21039216220378875732421875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.17117647826671600341796875, 0.61431372165679931640625, 0.70843136310577392578125, 0.469215691089630126953125, 0.1280392110347747802734375, 0.0339215695858001708984375, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.0652941167354583740234375, 0.092745102941989898681640625, 0.0417647063732147216796875, 0.02999999932944774627685546875, 0.02999999932944774627685546875, 0.02999999932944774627685546875}};

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


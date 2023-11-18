import random
from pathlib import Path

import numpy as np
import torch

CMD_TEMPLATE = """@esbmc @benchmark --quiet --force-malloc-success --no-bounds-check --no-div-by-zero-check --no-pointer-check --interval-analysis --fixedbv @HEADERS"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 7777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


DATA_FOLDER = Path.cwd().parent.joinpath("data")
MODELS_FOLDER = DATA_FOLDER.joinpath("networks")
PROPERTIES_FOLDER = DATA_FOLDER.joinpath("properties")

OUTPUT_FOLDER = Path.cwd().parent.joinpath("results")

_TMP = Path("/").joinpath("tmp")
ABSTRACTIONS_PATH = _TMP.joinpath("abstractions")
ABSTRACTIONS_PATH.mkdir(exist_ok=True)

ESBMC_ABSTRACTIONS_PATH = ABSTRACTIONS_PATH.joinpath("esbmc")
ESBMC_ABSTRACTIONS_PATH.mkdir(exist_ok=True)


ORIGINAL_FOLDER = _TMP.joinpath("original")
ORIGINAL_FOLDER.mkdir(exist_ok=True)

QUANTIZED_FOLDER = _TMP.joinpath("quantized")
QUANTIZED_FOLDER.mkdir(exist_ok=True)

BENCHMARKS_OPTIONS = [
    c.name.replace(".json", "") for c in PROPERTIES_FOLDER.glob("*.json")
]

GPFQ_MODELS = OUTPUT_FOLDER.joinpath("gpfq")

ONNX2C_PATH = Path.cwd().parent.joinpath("bin")
ESBMC_PATH = Path.cwd().parent.joinpath("bin")

ESBMC_PROPERTY_TEMPLATE = """
#include <stdlib.h>
#include <stdio.h>

#include "original.h"
#include "quantized.h"

#ifndef EQUIVALENCE
#define EQUIVALENCE @EQUIVALENCE
#endif

#ifndef EPSILON
#define EPSILON @EPSILON
#endif

#define BATCH 1

#ifndef CHANNELS
#define CHANNELS @CHANNELS
#endif

#ifndef HEIGHT
#define HEIGHT @HEIGHT
#endif

#ifndef WIDTH
#define WIDTH @WIDTH
#endif

#ifndef OUTPUT_SIZE
#define OUTPUT_SIZE @OUTPUT_SIZE
#endif

float nondet_float();

const float @LOWER_VECTOR = 
@LOWER_BOUNDS;

const float @UPPER_VECTOR = 
@UPPER_BOUNDS;

static inline void init_symbolic_input(float @INPUT_VECTOR)
{
    @INIT_SYMBOLIC
}

static inline void add_input_assumptions(float @INPUT_VECTOR)
{
    @ASSUMPTIONS
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
    __ESBMC_assert(property_holds, "Property Violated");
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

    __ESBMC_assert(property_holds, "Property Violated");
}

int main()
{

    // Define input and output vectors
    float @X_VECTOR;
    float original_output[BATCH][OUTPUT_SIZE];
    float quantized_output[BATCH][OUTPUT_SIZE];


    // Init symbolic input vector
    init_symbolic_input(x);

    // Add input assumptions
    add_input_assumptions(x);

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
"""

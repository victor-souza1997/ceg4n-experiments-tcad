import re
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ceg4n.constants import (
    ABSTRACTIONS_PATH,
    LAYER_H,
    ONNX2C_PATH,
    PROPERTY_ESBMC_C,
    PROPERTY_MAIN_C,
    Q_LAYERS_H,
    QUANTIZATION_H,
    UTILS_H,
)
from ceg4n.nn import ModelParams, load_model_parameters
from ceg4n.optimizer import CounterExample


def print_headers(benchmark):
    benchmark = benchmark.replace("-", "_")
    abs_path = ABSTRACTIONS_PATH.joinpath(benchmark)
    abs_path.mkdir(exist_ok=True)
    filename = abs_path.joinpath(f"layers.h")
    with open(str(filename), "w") as fp:
        fp.writelines(LAYER_H)

    filename = abs_path.joinpath(f"qlayers.h")
    with open(str(filename), "w") as fp:
        fp.writelines(Q_LAYERS_H)

    filename = abs_path.joinpath(f"quantization.h")
    with open(str(filename), "w") as fp:
        fp.writelines(QUANTIZATION_H)

    filename = abs_path.joinpath(f"utils.h")
    with open(str(filename), "w") as fp:
        fp.writelines(UTILS_H)


def print_benchmark(benchmark: str, model: ModelParams, suffix):
    suffix = f"{suffix}_" if suffix else ""
    benchmark = benchmark.replace("-", "_")
    abs_path = ABSTRACTIONS_PATH.joinpath(benchmark.lower())
    abs_path.mkdir(exist_ok=True)
    filename = abs_path.joinpath(abs_path, f"{suffix}{benchmark.lower()}.h")
    with open(filename, "w") as fp:
        # print(model.model)
        fp.writelines(f"\n#ifndef _{suffix.upper()}{benchmark.upper()}_H")
        fp.writelines(f"\n#define _{suffix.upper()}{benchmark.upper()}_H")
        fp.writelines('\n\n#include "string.h"')
        fp.writelines('\n\n#include "layers.h"')
        weight = [
            layer_id
            for layer_id, layer in model.quantizable_params.items()
            if isinstance(layer, nn.Linear)
        ]

        bias = [
            layer_id
            for layer_id, layer in model.quantizable_params.items()
            if isinstance(layer, nn.Linear)
        ]

        layers = [
            layer for _, layer in model.params.items() if isinstance(layer, nn.Linear)
        ]

        def formatted_array(arr):
            return ", ".join([str(a) for a in arr])

        for w in weight:
            layer = model.quantizable_params[w]
            w = w.replace("::", "").lower()
            fp.writelines(
                f"\n\nstatic const double {suffix}{w}_weight[{layer.out_features}][{layer.in_features}] ="
            )
            array = layer.weight.data.detach().cpu().numpy().tolist()
            fp.writelines("\n{")
            array = ",\n\t".join(["{ %s }" % formatted_array(arr) for arr in array])
            fp.writelines("\n\t%s" % array)
            fp.writelines("\n};")

        for b in bias:
            layer = model.quantizable_params[b]
            b = b.replace("::", "").lower()
            fp.writelines(
                f"\n\nstatic const double {suffix}{b}_bias[{layer.out_features}] ="
            )
            array = layer.bias.data.detach().cpu().numpy().tolist()
            fp.writelines("\n{")
            fp.writelines("\n\t%s" % formatted_array(array))
            fp.writelines("\n};")
        fp.writelines(
            f"\n\nstatic inline void {suffix}{benchmark.lower()}(const double *x, double *y)"
        )
        fp.writelines("\n{")

        previous_output = "x"
        previous_layer = None
        for layer_id, layer in model.params.items():
            layer_id = layer_id.replace("::", "").lower()
            if isinstance(layer, nn.Flatten):
                fp.writelines(f"\n\tdouble flatten_output[{layers[0].in_features}];")
                fp.writelines(
                    f"\n\tflatten((const double *) {previous_output}, (double *) flatten_output, {layers[0].in_features});"
                )
                previous_output = f"flatten_output"
            elif isinstance(layer, nn.Linear):
                fp.writelines(f"\n\tdouble {layer_id}_output[{layer.out_features}];")
                fp.writelines(
                    f"\n\tlinear((const double *) {layer_id}_weight, (const double *) {suffix}{layer_id}_bias, (const double *) {previous_output}, (double *) {layer_id}_output, {layer.out_features}, {layer.in_features});"
                )
                previous_output = f"{layer_id}_output"
                previous_layer = layer
            elif isinstance(layer, nn.ReLU):
                fp.writelines(
                    f"\n\tdouble {layer_id}_relu_output[{previous_layer.out_features}];"
                )
                fp.writelines(
                    f"\n\trelu((const double *) {previous_output}, (double *) {layer_id}_relu_output, {previous_layer.out_features});"
                )
                previous_output = f"{layer_id}_relu_output"

        fp.writelines(f"\n\tfor(size_t i = 0; i < {previous_layer.out_features}; i++)")
        fp.writelines("\n\t{")
        fp.writelines(f"\n\t\ty[i] = {previous_output}[i];")
        fp.writelines("\n\t}\n}")
        fp.writelines(f"\n\n#endif //_{benchmark.upper()}_H")


def print_quantized_benchmark(benchmark: str, model: ModelParams):
    benchmark = benchmark.replace("-", "_")
    abs_path = ABSTRACTIONS_PATH.joinpath(benchmark.lower())
    abs_path.mkdir(exist_ok=True)
    filename = abs_path.joinpath(abs_path, f"q_{benchmark.lower()}.h")
    with open(str(filename), "w") as fp:
        # print(model.model)
        q_benchmark = f"_Q_{benchmark.upper()}_H"
        fp.writelines(f"\n#ifndef {q_benchmark}")
        fp.writelines(f"\n#define {q_benchmark}")
        fp.writelines('\n\n#include "string.h"')
        fp.writelines('\n\n#include "layers.h"')

        weight = [
            layer_id
            for layer_id, layer in model.layers.items()
            if isinstance(layer, nn.Linear)
        ]

        bias = [
            layer_id
            for layer_id, layer in model.layers.items()
            if isinstance(layer, nn.Linear)
        ]

        layers = [
            layer for _, layer in model.layers.items() if isinstance(layer, nn.Linear)
        ]

        def formatted_array(arr):
            return ", ".join([str(a) for a in arr])

        for w in weight:
            layer = model.get_layer(w)
            fp.writelines(
                f"\n\nstatic const double Q_{w}_weight[{layer.out_features}][{layer.in_features}] ="
            )
            weight, _ = model.get_quantized_weights(w, quantization_bits)
            array = weight.detach().cpu().numpy().tolist()
            fp.writelines("\n{")
            array = ",\n\t".join(["{ %s }" % formatted_array(arr) for arr in array])
            fp.writelines("\n\t%s" % array)
            fp.writelines("\n};")

        for b in bias:
            layer = model.get_layer(b)
            fp.writelines(f"\n\nstatic const double Q_{b}_bias[{layer.out_features}] =")
            _, bias = model.get_quantized_weights(b, quantization_bits)
            array = bias.detach().cpu().numpy().tolist()
            fp.writelines("\n{")
            fp.writelines("\n\t%s" % formatted_array(array))
            fp.writelines("\n};")
        fp.writelines(
            f"\n\nstatic inline void q_{benchmark.lower()}(const double *x, double *y)"
        )
        fp.writelines("\n{")

        previous_output = "x"
        previous_layer = None
        for layer_id, layer in model.layers.items():
            if isinstance(layer, Flatten):
                fp.writelines(f"\n\tdouble flatten_output[{layers[0].in_features}];")
                fp.writelines(
                    f"\n\tflatten((const double *) {previous_output}, (double *) flatten_output, {layers[0].in_features});"
                )
                previous_output = "flatten_output"
            elif isinstance(layer, nn.Linear):
                fp.writelines(f"\n\tdouble {layer_id}_output[{layer.out_features}];")
                fp.writelines(
                    f"\n\tlinear((const double *) Q_{layer_id}_weight, (const double *) Q_{layer_id}_bias, (const double *) {previous_output}, (double *) {layer_id}_output, {layer.out_features}, {layer.in_features});"
                )
                previous_output = f"{layer_id}_output"
                previous_layer = layer
            elif isinstance(layer, nn.ReLU):
                fp.writelines(
                    f"\n\tdouble {layer_id}_relu_output[{previous_layer.out_features}];"
                )
                fp.writelines(
                    f"\n\trelu((const double *) {previous_output}, (double *) {layer_id}_relu_output, {previous_layer.out_features});"
                )
                previous_output = f"{layer_id}_relu_output"

        fp.writelines(f"\n\tfor(size_t i = 0; i < {previous_layer.out_features}; i++)")
        fp.writelines("\n\t{")
        fp.writelines(f"\n\t\ty[i] = {previous_output}[i];")
        fp.writelines("\n\t}\n}")

        fp.writelines(f"\n\n#endif //{q_benchmark}")
        return previous_layer.out_features


def print_esbmc(benchmark: str, ce: CounterExample):
    benchmark = benchmark.replace("-", "_")
    abs_path = ABSTRACTIONS_PATH.joinpath(benchmark.lower())
    abs_path.mkdir(exist_ok=True)
    labels = [ce.y]
    # checker = "argmin" if spec.classifier_fn == torch.argmin else "argmax"
    checker = "argmax"
    filename = abs_path.joinpath(f"esbmc_{benchmark.lower()}_{ce.y}.c")
    _benchmark = benchmark.lower()
    q_benchmark = f"q_{_benchmark}"
    _features = ce.x
    _features_size = len(_features)
    _specs_size = len(labels)
    filecontent = PROPERTY_ESBMC_C.replace("@benchmark", _benchmark)
    filecontent = filecontent.replace("@qbenchmark", q_benchmark)
    filecontent = filecontent.replace("@features_size", str(_features_size))
    filecontent = filecontent.replace("@output_size", str(ce.output_size))
    filecontent = filecontent.replace("@specs_size", str(_specs_size))
    filecontent = filecontent.replace("@specs", ",".join([str(f) for f in labels]))
    filecontent = filecontent.replace(
        "@features", ",".join([str(f) for f in _features])
    )
    filecontent = filecontent.replace("@checker", checker)
    # filecontent = filecontent.replace("@eps", str(eps))
    with open(str(filename), "w") as fp:
        fp.writelines(filecontent)


def print_main(benchmark: str, counter_example: CounterExample):
    benchmark = benchmark.replace("-", "_")
    abs_path = ABSTRACTIONS_PATH.joinpath(benchmark.lower())
    abs_path.mkdir(exist_ok=True)

    labels = [counter_example.y]
    # checker = "argmin" if spec.classifier_fn == torch.argmin else "argmax"
    checker = "argmax"
    filename = abs_path.joinpath(f"main_{benchmark.lower()}_{counter_example.y}.c")
    _benchmark = benchmark.lower()
    q_benchmark = f"q_{_benchmark}"
    _features = counter_example.x
    _features_size = len(_features)
    _specs_size = len(labels)
    filecontent = PROPERTY_MAIN_C.replace("@benchmark", _benchmark)
    filecontent = filecontent.replace("@qbenchmark", q_benchmark)
    filecontent = filecontent.replace("@features_size", str(_features_size))
    filecontent = filecontent.replace("@output_size", str(counter_example.output_size))
    filecontent = filecontent.replace("@specs_size", str(_specs_size))
    filecontent = filecontent.replace("@specs", ",".join([str(f) for f in labels]))
    filecontent = filecontent.replace(
        "@features", ",".join([str(f) for f in _features])
    )
    filecontent = filecontent.replace("@checker", checker)
    with open(filename, "w") as fp:
        fp.writelines(filecontent)


def generate_abstractions(
    model_file: str,
    quant_model_file: str,
    specs: List[CounterExample],
    # individual: List[int]
):
    benchmark = model_file.split("/")[-1].split(".")[0]
    print_headers(benchmark=benchmark)
    print_benchmark(benchmark, load_model_parameters(model_file), suffix=None)
    print_benchmark(benchmark, load_model_parameters(model_file), suffix="q")
    # print_quantized_benchmark(benchmark, load_model_parameters(quant_model_file))
    for ce in specs:
        print_main(benchmark, ce)
        print_esbmc(benchmark, ce)
        #
    pass
    #
    #
    #
    #


# def generate_abstractions_v2(
#     benchmark: str,
#     model: QuantizableModule,
#     qmodel: QuantizableModule,
#     specs: List[CounterExample]
# ):
#     print_headers(benchmark=benchmark)
#     print_benchmark(benchmark, model)
#     output_size = print_quantized_benchmark(benchmark, None, qmodel)
#     print_esbmc(benchmark, specs, output_size)
#     print_main(benchmark, specs, output_size)

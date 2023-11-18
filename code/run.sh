#!/bin/bash

poetry install

poetry run main \
    --benchmark iris_4x2 \
    --optimizer NSGAII \
    --verifier ESBMC \
    --equivalence TOP \
    --size 0.03

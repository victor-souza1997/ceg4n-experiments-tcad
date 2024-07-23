# rq-cegio-experiments

## Setup

```bash
# ~/ceg4n-experiments
./setup.sh
```


## Add ESBMC binary to the path
```bash
# ~/ceg4n-experiments
export ONNX2C_PATH=$(pwd)/bin \
    && export PATH="${ONNX2C_PATH}:${PATH}" \
    && export ESBMC_PATH=$(pwd)/bin \
    && export PATH="${ESBMC_PATH}/bin:${PATH}" 
```

## Run CEG4N

### Using ESBMC
```bash
# ~/ceg4n-experiments/code
cd code \
    && poetry run main \
    --benchmark iris_4x2 \
    --optimizer NSGAII \
    --verifier ESBMC \
    --equivalence TOP \
    --size 0.03 \
    && cd ..
```

### Using NNEquiv
```bash
# ~/code
poetry run main \
    --benchmark iris_4x2 \
    --optimizer NSGAII \
    --verifier NNEQUIV \
    --equivalence TOP \
    --size 0.03
```

## Objective

- [ ] Update ESBMC version.
- [ ] Add new database to the neural network.
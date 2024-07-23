import onnx
from seeds import *
# Load the ONNX model
import onnxruntime as ort
import numpy as np
import torch

# Load the ONNX model
model_path = 'seeds_2x1.onnx'
session = ort.InferenceSession(model_path)

# Prepare your test data
test = seeds_dataloaders()['train']
print(test)
test_data_tensor = torch.tensor([[11.36,13.05,0.8382,5.175,2.755,4.048,5.263]])
test_data_numpy = test_data_tensor.numpy()

input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: test_data_numpy})

# Print the result
output = result
output = np.argmax(output[0]), output[0]
print("Model output:", output)
# Print the result
#output = result[0]
#print("Model output:", output)

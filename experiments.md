# Quantization efficiency metrics

- Inference Time
  - Estimate MACs (Multiply-Accumulate-Operation)
    - Same as FP Model, but with integer arithmetic
  - Actual Hardware Integer Operations/FLOPS
    - Not measurable, highly backend depending
  - Actual Time
    - Feasible when using the same backend
    - Not directly comparable to the FP Model, due to backend differences

- Memory
  - Estimated Size (bits * params)
  - Stored State-Dict size


# Quantization Customization
- Weight Observers
  - up to 7 Bits
  - Range:
    - full range [-64, 63]
    - symmetric range [-63, 63]
    - non-negative range [0, 63]
  - Statistics:
    - Min/Max
  - Granularity
    - Per Channel
    - Per Tensor
- Activation Observers
  - up to 7 Bits
  - Range:
    - non-negative range [0, 127]
  - Statistics (static):
    - Min/Max
    - Min/Max (moving avg.)
    - Histogramm
  - Granularity:
    - Per Tensor
  - Dynamic (only for linear layers)
- Mixed Quantization
  - Conv
    - Static
    - No Quantization
  - Linear
    - Static
    - Dynamic
    - No Quantization
- PTQ
  - limit calibration set
- QAT
  - optimizer + settings
  - stop criterion

**$\to$ 2700 Quantization Configurations, each can be PTQ or QAT**

# PTQ

- Static Quantization
  - [x] Different Observers
  - [x] Different Quantization Ranges
- Mixed Dynamic Quantization
  - [ ] Dynamic Quantization of linear layers, No quantization of conv layers
  - [ ] Dynamic Quantization of linear layers, static quantization of conv layers

# QAT

- Static Quantization
  - [x] Different Observers
  - [x] Different Quantization Ranges
- Mixed Dynamic Quantization
  - [ ] Dynamic Quantization of linear layers, No quantization of conv layers
  - [ ] Dynamic Quantization of linear layers, static quantization of conv layers

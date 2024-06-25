# Raspberry Pi Pico GPIO State Inference with Neural Networks

This repository demonstrates a Deep Neural Network inference model created in TensorFlow, converted to a TensorFlow Lite for Microcontrollers model, and run on a Raspberry Pi Pico. The model is designed to analyze the data of incoming button clicks and classify them into three states: single click, double click, and no action. The training data for each state is a time series of button operation states sampled at 32 Hz, with only 10 events per condition.

## Features

- Classifies GPIO button states into single click, double click, and no action
- Uses a simple neural network model implemented in TensorFlow
- Converts the model to TensorFlow Lite micro format for use on microcontrollers
- Runs inference on a Raspberry Pi Pico

## Build and Install

### 1. Set up Python Environment

Install the necessary Python libraries, create a model, and train it with the training data. The created model file will be a TensorFlow Lite model (`model.tflite`) and a model embedded in a C header (`model.h`):

```bash
pip install -r requirements.txt
python create_model.py
```

### 2. Build the Firmware

Build the firmware for the Raspberry Pi Pico that performs the inference:

```bash
git submodule update --init
mkdir build && cd build
PICO_SDK_PATH=/path/to/pico-sdk cmake ..
make
```
If the build is successful, the `neuralclick.uf2` firmware will be created. Drag and drop it onto the Raspberry Pi Pico running in BOOTSEL mode to install.

### 3. Running the Inference

Inference status is output to USB CDC and UART. Connect to the Raspberry Pi Pico using a serial terminal to see the output.


## Example

### Sample Input

The input is a time series of GPIO states sampled at 32 Hz. For example:

```python
[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Single click
[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]   # Double click
```

### Sample Output

The output will classify the input into one of the three states:

```
Nop 0.000, Single click 0.250, Double click 0.750 -> Predicted label: Double click
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.md) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro)
- [Raspberry Pi Pico](https://www.raspberrypi.com/products/raspberry-pi-pico/)
- [Raspberry Pi Pico SDK](https://github.com/raspberrypi/pico-sdk)

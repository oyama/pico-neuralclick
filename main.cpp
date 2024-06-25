/*
 * Inference of GPIO button input states using TensorFlow Lite models
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include <hardware/gpio.h>
#include <hardware/structs/ioqspi.h>
#include <hardware/sync.h>
#include <pico/stdlib.h>
#include <stdio.h>

#include "model.h"

#define SAMPLE_RATE_HZ 32
#define SAMPLE_PERIOD_MS (1000 / SAMPLE_RATE_HZ)
#define NUM_SAMPLES 20

bool __no_inline_not_in_flash_func(bb_get_bootsel_button)() {
    const uint CS_PIN_INDEX = 1;
    uint32_t flags = save_and_disable_interrupts();
    hw_write_masked(&ioqspi_hw->io[CS_PIN_INDEX].ctrl,
                    GPIO_OVERRIDE_LOW << IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_LSB,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_BITS);
    for (volatile int i = 0; i < 1000; ++i);
    bool button_state = !(sio_hw->gpio_hi_in & (1u << CS_PIN_INDEX));
    hw_write_masked(&ioqspi_hw->io[CS_PIN_INDEX].ctrl,
                    GPIO_OVERRIDE_NORMAL << IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_LSB,
                    IO_QSPI_GPIO_QSPI_SS_CTRL_OEOVER_BITS);
    restore_interrupts(flags);

    return button_state;
}

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    int inference_count = 0;

    constexpr int kTensorArenaSize = 1024 * 64;
    uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  tflite::InitializeTarget();
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED());
  resolver.AddSoftmax(tflite::Register_SOFTMAX());
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  inference_count = 0;
}

int main(void) {
    stdio_init_all();
    setup();

    float samples[NUM_SAMPLES] = {0};
    while (true) {
        for (int i = NUM_SAMPLES - 1; i > 0; i--) {
            samples[i] = samples[i - 1];
        }
        samples[0] = bb_get_bootsel_button() ? 1.0f : 0.0f;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            input->data.f[i] = samples[i];
        }

        if (interpreter->Invoke() != kTfLiteOk) {
            printf("Invoke failed.");
            return 1;
        }

        int predicted_index = 0;
        float max_value = output->data.f[0];
        for (int i = 0; i < output->dims->data[1]; i++) {
            if (output->data.f[i] > max_value) {
                max_value = output->data.f[i];
                predicted_index = i;
            }
        }
        const char* labels[] = {"Nop", "Single click", "Double click"};
        printf("Nop %.2f, Single click %.2f, Double click %.2f -> Predicted label: %s\n",
               output->data.f[0], output->data.f[1], output->data.f[2],
               labels[predicted_index]);

        sleep_ms(SAMPLE_PERIOD_MS);
    }
}


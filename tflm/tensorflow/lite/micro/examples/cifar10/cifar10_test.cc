/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_float_model.h"
#include "tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_int8_model.h"
#include "tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_pruned_int8_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
//for debug
#include <iostream>

// added by i.jeong
// code for load cifar10 data
#define DATA_IDX 0
#define NUM_CLASS 10
#define NUM_SAMPLES 100
#define test_data_path "/workspace/tflm/tflite-micro/tensorflow/lite/micro/examples/cifar10/data/test_batch.bin"
#include "tensorflow/lite/micro/examples/cifar10/cifar10_loader.h"

CIFAR10Loader loader;
auto data = loader.loadFile(test_data_path);

namespace {

// added by i.jeong
// change op_resolver size 
using OpResolver = tflite::MicroMutableOpResolver<12>;

TfLiteStatus RegisterOps(OpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));
  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_float_tflite), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1);
  interpreter.input(0)->data.f[0] = 1.f;
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Added by i.jeong
  // setup input and perform inference
  auto img = loader.getImage(DATA_IDX);

  // // need to rearrange the input data
  // // need to change rrr ggg bbb -> rgb rgb rgb
  // uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 
  // for (uint16_t i = 0; i < num_pixels; i++){
  //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 0] = img.data[i] / 255.0;
  //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 1] = img.data[num_pixels + i] / 255.0;
  //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 2] = img.data[num_pixels * 2 + i] / 255.0;
  // }
  // NCHW (rrr, ggg, bbb)
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 
  for (uint16_t i = 0; i < num_pixels; i++) {
      interpreter.input(0)->data.f[i] = img.data[i] / 255.0;                  
      interpreter.input(0)->data.f[num_pixels + i] = img.data[num_pixels + i] / 255.0; 
      interpreter.input(0)->data.f[num_pixels * 2 + i] = img.data[num_pixels * 2 + i] / 255.0; 
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  uint8_t y_pred = std::distance(interpreter.output(0)->data.f, std::max_element(interpreter.output(0)->data.f, interpreter.output(0)->data.f + NUM_CLASS));
  MicroPrintf("[float model] y_pred: %d", y_pred);
  MicroPrintf("[float model] y_real: %d\n", (int)(img.label));

  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInferenceForAllData() {
  const tflite::Model* model =
      ::tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_float_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Added by i.jeong
  // setup input and perform inference
  double accuracy = 0.0;
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 
  // const float mean[3] = {0.4914, 0.4822, 0.4465};
  // const float std[3] = {0.2023, 0.1994, 0.2010};
  //counter list length is 10, list of zeros
  std::vector<std::vector<uint8_t>> counter_list(10);

  for (uint16_t data_idx = 0; data_idx < loader.size(); data_idx++){
    auto img = loader.getImage(data_idx);

    // // need to rearrange the input data
    // // need to change rrr ggg bbb -> rgb rgb rgb
    // for (uint16_t i = 0; i < num_pixels; i++){
    //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 0] = static_cast<float>((img.data[i]) / 255.0f);
    //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 1] = static_cast<float>((img.data[num_pixels + i]) / 255.0f);
    //   interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 2] = static_cast<float>((img.data[num_pixels * 2 + i]) / 255.0f);
    //   // interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 0] = (static_cast<float>((img.data[i]) / 255.0f) - mean[0]) / std[0];
    //   // interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 1] = (static_cast<float>((img.data[num_pixels + i]) / 255.0f) - mean[1]) / std[1];
    //   // interpreter.input(0)->data.f[i*loader.NUM_CHANNELS + 2] = (static_cast<float>((img.data[num_pixels * 2 + i]) / 255.0f) - mean[2]) / std[2];
    // }
    // NCHW (rrr, ggg, bbb)
    for (uint16_t i = 0; i < num_pixels; i++) {
        interpreter.input(0)->data.f[i] = img.data[i] / 255.0;                  
        interpreter.input(0)->data.f[num_pixels + i] = img.data[num_pixels + i] / 255.0; 
        interpreter.input(0)->data.f[num_pixels * 2 + i] = img.data[num_pixels * 2 + i] / 255.0;
    }

    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    
    uint8_t y_pred = std::distance(interpreter.output(0)->data.f, std::max_element(interpreter.output(0)->data.f, interpreter.output(0)->data.f + NUM_CLASS));
    if (y_pred == (uint8_t)img.label){
      accuracy += 1.0;
    }
   
  }
  accuracy /= loader.size();
  MicroPrintf("[Counter List]:", counter_list);
  MicroPrintf("[float model] accuracy: %f", accuracy);

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size might need adjustment for int8 model
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                     kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Get quantization parameters for input tensor
  TfLiteTensor* input = interpreter.input(0);
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Get quantization parameters for output tensor
  TfLiteTensor* output = interpreter.output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  MicroPrintf("[int8 model] input_scale: %f", static_cast<double>(input_scale));
  MicroPrintf("[int8 model] input_zero_point: %d", input_zero_point);
  MicroPrintf("[int8 model] output_scale: %f", static_cast<double>(output_scale));
  MicroPrintf("[int8 model] output_zero_point: %d", output_zero_point);

  auto img = loader.getImage(DATA_IDX);
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE; 
  // NCHW format (rrr, ggg, bbb)
  for (uint16_t i = 0; i < num_pixels; i++) {
    // Convert [0, 255] -> [0, 1] -> int8 quantized value
    float normalized_r = img.data[i] / 255.0f;
    float normalized_g = img.data[num_pixels + i] / 255.0f;
    float normalized_b = img.data[num_pixels * 2 + i] / 255.0f;

    // Apply quantization: q = (f / scale) + zero_point
    input->data.int8[i] = static_cast<int8_t>(normalized_r / input_scale + input_zero_point);
    input->data.int8[num_pixels + i] = static_cast<int8_t>(normalized_g / input_scale + input_zero_point);
    input->data.int8[num_pixels * 2 + i] = static_cast<int8_t>(normalized_b / input_scale + input_zero_point);
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  // Find predicted class from quantized output
  // First dequantize output values
  std::vector<float> dequantized_output(NUM_CLASS);
  for (int i = 0; i < NUM_CLASS; i++) {
    // Dequantization: f = (q - zero_point) * scale
    dequantized_output[i] = (output->data.int8[i] - output_zero_point) * output_scale;
  }

  // Find max element index
  uint8_t y_pred = std::distance(dequantized_output.begin(), 
                                std::max_element(dequantized_output.begin(), dequantized_output.end()));
  MicroPrintf("[int8 model] y_pred: %d", y_pred);
  MicroPrintf("[int8 model] y_real: %d\n", (int)(img.label));

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInferenceForAllData() {
  const tflite::Model* model =
      ::tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size might need adjustment for int8 model
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                     kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Get quantization parameters for input tensor
  TfLiteTensor* input = interpreter.input(0);
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Get quantization parameters for output tensor
  TfLiteTensor* output = interpreter.output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  MicroPrintf("[int8 model] input_scale: %f", static_cast<double>(input_scale));
  MicroPrintf("[int8 model] input_zero_point: %d", input_zero_point);
  MicroPrintf("[int8 model] output_scale: %f", static_cast<double>(output_scale));
  MicroPrintf("[int8 model] output_zero_point: %d", output_zero_point);

  double accuracy = 0.0;
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE;

  for (uint16_t data_idx = 0; data_idx < loader.size(); data_idx++) {
    auto img = loader.getImage(data_idx);

    // Quantize input data from uint8 [0-255] to int8 using input quantization parameters
    // NCHW format (rrr, ggg, bbb)
    for (uint16_t i = 0; i < num_pixels; i++) {
      // Convert [0, 255] -> [0, 1] -> int8 quantized value
      float normalized_r = img.data[i] / 255.0f;
      float normalized_g = img.data[num_pixels + i] / 255.0f;
      float normalized_b = img.data[num_pixels * 2 + i] / 255.0f;

      // Apply quantization: q = (f / scale) + zero_point
      input->data.int8[i] = static_cast<int8_t>(normalized_r / input_scale + input_zero_point);
      input->data.int8[num_pixels + i] = static_cast<int8_t>(normalized_g / input_scale + input_zero_point);
      input->data.int8[num_pixels * 2 + i] = static_cast<int8_t>(normalized_b / input_scale + input_zero_point);
    }

    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    // Find predicted class from quantized output
    // First dequantize output values
    std::vector<float> dequantized_output(NUM_CLASS);
    for (int i = 0; i < NUM_CLASS; i++) {
      // Dequantization: f = (q - zero_point) * scale
      dequantized_output[i] = (output->data.int8[i] - output_zero_point) * output_scale;
    }

    // Find max element index
    uint8_t y_pred = std::distance(dequantized_output.begin(), 
                                 std::max_element(dequantized_output.begin(), dequantized_output.end()));
    
    if (y_pred == (uint8_t)img.label) {
      accuracy += 1.0;
    }
  }

  accuracy /= loader.size();
  MicroPrintf("[int8 model] accuracy: %f", accuracy);

  return kTfLiteOk;
}

TfLiteStatus LoadPrunedQuantModelAndPerformInferenceForAllData() {
  const tflite::Model* model =
      ::tflite::GetModel(tensorflow_lite_micro_examples_cifar10_models_cifar10_pruned_int8_tflite);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size might need adjustment for int8 model
  constexpr int kTensorArenaSize = 550000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                     kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Get quantization parameters for input tensor
  TfLiteTensor* input = interpreter.input(0);
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  // Get quantization parameters for output tensor
  TfLiteTensor* output = interpreter.output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  MicroPrintf("[int8 model] input_scale: %f", static_cast<double>(input_scale));
  MicroPrintf("[int8 model] input_zero_point: %d", input_zero_point);
  MicroPrintf("[int8 model] output_scale: %f", static_cast<double>(output_scale));
  MicroPrintf("[int8 model] output_zero_point: %d", output_zero_point);

  double accuracy = 0.0;
  uint16_t num_pixels = loader.IMAGE_SIZE * loader.IMAGE_SIZE;

  for (uint16_t data_idx = 0; data_idx < loader.size(); data_idx++) {
    auto img = loader.getImage(data_idx);

    // Quantize input data from uint8 [0-255] to int8 using input quantization parameters
    // NCHW format (rrr, ggg, bbb)
    for (uint16_t i = 0; i < num_pixels; i++) {
      // Convert [0, 255] -> [0, 1] -> int8 quantized value
      float normalized_r = img.data[i] / 255.0f;
      float normalized_g = img.data[num_pixels + i] / 255.0f;
      float normalized_b = img.data[num_pixels * 2 + i] / 255.0f;

      // Apply quantization: q = (f / scale) + zero_point
      input->data.int8[i] = static_cast<int8_t>(normalized_r / input_scale + input_zero_point);
      input->data.int8[num_pixels + i] = static_cast<int8_t>(normalized_g / input_scale + input_zero_point);
      input->data.int8[num_pixels * 2 + i] = static_cast<int8_t>(normalized_b / input_scale + input_zero_point);
    }

    TF_LITE_ENSURE_STATUS(interpreter.Invoke());

    // Find predicted class from quantized output
    // First dequantize output values
    std::vector<float> dequantized_output(NUM_CLASS);
    for (int i = 0; i < NUM_CLASS; i++) {
      // Dequantization: f = (q - zero_point) * scale
      dequantized_output[i] = (output->data.int8[i] - output_zero_point) * output_scale;
    }

    // Find max element index
    uint8_t y_pred = std::distance(dequantized_output.begin(), 
                                 std::max_element(dequantized_output.begin(), dequantized_output.end()));
    
    if (y_pred == (uint8_t)img.label) {
      accuracy += 1.0;
    }
  }

  accuracy /= loader.size();
  MicroPrintf("[pruned int8 model] accuracy: %f", accuracy);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  tflite::InitializeTarget();

  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  
  // function call for inference of one sample
  // MicroPrintf("\n~~~INFERENCE OF ONE SAMPLE~~~\n");
  // TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  // TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  // TF_LITE_ENSURE_STATUS(LoadPrunedQuantModelAndPerformInference());

  // function calll for inference of all samples
  MicroPrintf("\n~~~INFERENCE OF ALL SAMPLES~~~\n");
  TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInferenceForAllData());
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInferenceForAllData());
  TF_LITE_ENSURE_STATUS(LoadPrunedQuantModelAndPerformInferenceForAllData());

  MicroPrintf("\n~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}

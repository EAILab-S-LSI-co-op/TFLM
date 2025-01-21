# # cifar-10
# CIFAR10_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/CIFAR10.pb"
# CIFAR10_PRUNED_05_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/CIFAR10-pruned-0.5.pb"

# # quantization: build ptq
# bazel build tensorflow/lite/micro/examples/cifar10/quantization:ptq 
# bazel-bin/tensorflow/lite/micro/examples/cifar10/quantization/ptq\
#    --source_model_dir=$CIFAR10_TF_MODEL_DIR \
#    --target_dir=tensorflow/lite/micro/examples/cifar10/models\
#    --model_name=cifar10_int8
# bazel-bin/tensorflow/lite/micro/examples/cifar10/quantization/ptq\
#    --source_model_dir=$CIFAR10_PRUNED_05_TF_MODEL_DIR \
#    --target_dir=tensorflow/lite/micro/examples/cifar10/models\
#    --model_name=cifar10_int8_pruned


# fmnist
FMNIST_ROOT="/workspace/tflm/tflite-micro-codes-complete/tflite-without-normalization"
FMNIST_TF_MODEL_DIR="$FMNIST_ROOT/tf/FMNIST.pb"
FMNIST_TF_PRUNED_MODEL_DIR="$FMNIST_ROOT/tf/FMNIST-pruned-0.5.pb"
FMNIST_TFLITE_MODEL_PATH="$FMNIST_ROOT/tflite/FMNIST.tflite"

# cp fmnist float to tflm models
FMNIST_TFLM_ROOT="tensorflow/lite/micro/examples/fmnist"
cp $FMNIST_TFLITE_MODEL_PATH "$FMNIST_TFLM_ROOT/models/fmnist_float.tflite"


FMNIST_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite-without-normalization/tf/FMNIST.pb"
FMNIST_PRUNED_05_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite-without-normalization/tf/FMNIST-pruned-0.5.pb"
FMNIST_TFLITE_MODEL_PATH="/workspace/tflm/tflite-micro-codes-complete/tflite-without-normalization/tflite/FMNIST.tflite"

# quantization: build ptq
bazel build tensorflow/lite/micro/examples/fmnist/quantization:ptq 
bazel-bin/tensorflow/lite/micro/examples/fmnist/quantization/ptq\
   --source_model_dir=$FMNIST_TF_MODEL_DIR \
   --target_dir=tensorflow/lite/micro/examples/fmnist/models\
   --model_name=fmnist_int8
bazel-bin/tensorflow/lite/micro/examples/fmnist/quantization/ptq\
   --source_model_dir=$FMNIST_PRUNED_05_TF_MODEL_DIR \
   --target_dir=tensorflow/lite/micro/examples/fmnist/models\
   --model_name=fmnist_int8_pruned

FMNIST_TFLITE_INT8_MODEL_PATH="tensorflow/lite/micro/examples/fmnist/models/fmnist_int8.tflite"
FMNIST_TFLITE_PRUNED_INT8_MODEL_PATH="tensorflow/lite/micro/examples/fmnist/models/fmnist_int8_pruned.tflite"
FMNIST_TFLITE_FLOAT_MODEL_PATH="tensorflow/lite/micro/examples/fmnist/models/fmnist_float.tflite"
cp $FMNIST_TFLITE_MODEL_PATH $FMNIST_TFLITE_FLOAT_MODEL_PATH

FMNIST_TFLITE_INT8_C_PATH="tensorflow/lite/micro/examples/fmnist/models/generated_fmnist_int8_model.h"
FMNIST_TFLITE_PRUNED_INT8_C_PATH="tensorflow/lite/micro/examples/fmnist/models/generated_fmnist_pruned_int8_model.h"
FMNIST_TFLITE_FLOAT_C_PATH="tensorflow/lite/micro/examples/fmnist/models/generated_fmnist_float_model.h"

xxd -i $FMNIST_TFLITE_INT8_MODEL_PATH > $FMNIST_TFLITE_INT8_C_PATH
xxd -i $FMNIST_TFLITE_PRUNED_INT8_MODEL_PATH > $FMNIST_TFLITE_PRUNED_INT8_C_PATH
xxd -i $FMNIST_TFLITE_FLOAT_MODEL_PATH > $FMNIST_TFLITE_FLOAT_C_PATH

make -f tensorflow/lite/micro/tools/make/Makefile test_fmnist_test

# mnist
MNIST_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/MNIST.pb"
MNIST_PRUNED_05_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/MNIST-pruned-0.5.pb"
MNIST_TFLITE_MODEL_PATH="/workspace/tflm/tflite-micro-codes-complete/tflite/tflite/MNIST.tflite"

# quantization: build ptq
bazel build tensorflow/lite/micro/examples/mnist/quantization:ptq 
bazel-bin/tensorflow/lite/micro/examples/mnist/quantization/ptq\
   --source_model_dir=$MNIST_TF_MODEL_DIR \
   --target_dir=tensorflow/lite/micro/examples/mnist/models\
   --model_name=mnist_int8
bazel-bin/tensorflow/lite/micro/examples/mnist/quantization/ptq\
   --source_model_dir=$MNIST_PRUNED_05_TF_MODEL_DIR \
   --target_dir=tensorflow/lite/micro/examples/mnist/models\
   --model_name=mnist_int8_pruned

MNIST_TFLITE_INT8_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_int8.tflite"
MNIST_TFLITE_PRUNED_INT8_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_int8_pruned.tflite"
MNIST_TFLITE_FLOAT_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_float.tflite"
cp $MNIST_TFLITE_MODEL_PATH  MNIST_TFLITE_FLOAT_MODEL_PATH

MNIST_TFLITE_INT8_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_int8_model.h"
MNIST_TFLITE_PRUNED_INT8_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_pruned_int8_model.h"
MNIST_TFLITE_FLOAT_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_float_model.h"

xxd -i $MNIST_TFLITE_INT8_MODEL_PATH > $MNIST_TFLITE_INT8_C_PATH
xxd -i $MNIST_TFLITE_PRUNED_INT8_MODEL_PATH > $MNIST_TFLITE_PRUNED_INT8_C_PATH
xxd -i $MNIST_TFLITE_FLOAT_MODEL_PATH > $MNIST_TFLITE_FLOAT_C_PATH

make -f tensorflow/lite/micro/tools/make/Makefile test_mnist_test

# ~~~INFERENCE OF ONE SAMPLE~~~

# [float model] y_pred: 2
# [float model] y_real: 2

# [int8 model] y_pred: 2
# [int8 model] y_real: 2

# [pruned int8 model] y_pred: 2
# [pruned int8 model] y_real: 2


# ~~~INFERENCE OF ALL SAMPLES~~~

# [float model] accuracy: 0.991800
# [int8 model] accuracy: 0.971200
# [pruned int8 model] accuracy: 0.970900

# ~~~ALL TESTS PASSED~~~

# Running mnist_test took 291.690 seconds
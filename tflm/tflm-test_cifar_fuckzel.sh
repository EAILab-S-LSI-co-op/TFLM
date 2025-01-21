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


# cifar10
TFLITE_MODEL_ROOT="/workspace/tflm/tflite-micro-codes-complete/tflite-without-normalization"
# CIFAR10_TF_MODEL_DIR="$TFLITE_MODEL_ROOT/tf/CIFAR10.pb"
# CIFAR10_PRUNED_05_TF_MODEL_DIR="$TFLITE_MODEL_ROOT/tf/cifar10-pruned-0.5.pb"
CIFAR10_TFLITE_MODEL_PATH="$TFLITE_MODEL_ROOT/tflite/CIFAR10.tflite"
CIFAR10_TFLM_MODEL_ROOT="/workspace/tflm/tflite-micro-codes-complete/tflm/tflite-micro/tensorflow/lite/micro/examples/cifar10/models"
cp $CIFAR10_TFLITE_MODEL_PATH  $CIFAR10_TFLM_MODEL_ROOT/cifar10_float.tflite
CIFAR10_TFLITE_FLOAT_MODEL_PATH="$CIFAR10_TFLM_MODEL_ROOT/cifar10_float.tflite"


CIFAR10_TFLITE_INT8_MODEL_PATH="tensorflow/lite/micro/examples/cifar10/models/cifar10_int8.tflite"
CIFAR10_TFLITE_PRUNED_INT8_MODEL_PATH="tensorflow/lite/micro/examples/cifar10/models/cifar10_int8_pruned.tflite"
CIFAR10_TFLITE_FLOAT_MODEL_PATH="tensorflow/lite/micro/examples/cifar10/models/cifar10_float.tflite"
cp $CIFAR10_TFLITE_MODEL_PATH $CIFAR10_TFLITE_FLOAT_MODEL_PATH

CIFAR10_TFLITE_INT8_C_PATH="tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_int8_model.h"
CIFAR10_TFLITE_PRUNED_INT8_C_PATH="tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_pruned_int8_model.h"
CIFAR10_TFLITE_FLOAT_C_PATH="tensorflow/lite/micro/examples/cifar10/models/generated_cifar10_float_model.h"

xxd -i $CIFAR10_TFLITE_INT8_MODEL_PATH > $CIFAR10_TFLITE_INT8_C_PATH
xxd -i $CIFAR10_TFLITE_PRUNED_INT8_MODEL_PATH > $CIFAR10_TFLITE_PRUNED_INT8_C_PATH
xxd -i $CIFAR10_TFLITE_FLOAT_MODEL_PATH > $CIFAR10_TFLITE_FLOAT_C_PATH

make -f tensorflow/lite/micro/tools/make/Makefile test_cifar10_test

# # mnist
# MNIST_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/MNIST.pb"
# MNIST_PRUNED_05_TF_MODEL_DIR="/workspace/tflm/tflite-micro-codes-complete/tflite/tf/MNIST-pruned-0.5.pb"
# MNIST_TFLITE_MODEL_PATH="/workspace/tflm/tflite-micro-codes-complete/tflite/tflite/MNIST.tflite"

# # quantization: build ptq
# bazel build tensorflow/lite/micro/examples/mnist/quantization:ptq 
# bazel-bin/tensorflow/lite/micro/examples/mnist/quantization/ptq\
#    --source_model_dir=$MNIST_TF_MODEL_DIR \
#    --target_dir=tensorflow/lite/micro/examples/mnist/models\
#    --model_name=mnist_int8
# bazel-bin/tensorflow/lite/micro/examples/mnist/quantization/ptq\
#    --source_model_dir=$MNIST_PRUNED_05_TF_MODEL_DIR \
#    --target_dir=tensorflow/lite/micro/examples/mnist/models\
#    --model_name=mnist_int8_pruned

# MNIST_TFLITE_INT8_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_int8.tflite"
# MNIST_TFLITE_PRUNED_INT8_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_int8_pruned.tflite"
# MNIST_TFLITE_FLOAT_MODEL_PATH="tensorflow/lite/micro/examples/mnist/models/mnist_float.tflite"
# cp $MNIST_TFLITE_MODEL_PATH  MNIST_TFLITE_FLOAT_MODEL_PATH

# MNIST_TFLITE_INT8_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_int8_model.h"
# MNIST_TFLITE_PRUNED_INT8_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_pruned_int8_model.h"
# MNIST_TFLITE_FLOAT_C_PATH="tensorflow/lite/micro/examples/mnist/models/generated_mnist_float_model.h"

# xxd -i $MNIST_TFLITE_INT8_MODEL_PATH > $MNIST_TFLITE_INT8_C_PATH
# xxd -i $MNIST_TFLITE_PRUNED_INT8_MODEL_PATH > $MNIST_TFLITE_PRUNED_INT8_C_PATH
# xxd -i $MNIST_TFLITE_FLOAT_MODEL_PATH > $MNIST_TFLITE_FLOAT_C_PATH

# make -f tensorflow/lite/micro/tools/make/Makefile test_mnist_test

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
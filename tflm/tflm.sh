TFLITE_ROOT="/workspace/tflm/tflite-micro-codes-complete/tflite"
CIFAR10_TFLM_PATH="tensorflow/lite/micro/examples/cifar10"


# cifar10
CIFAR10_TFLITE_FLOAT_MODEL_PATH="$TFLITE_ROOT/tflite/CIFAR10.tflite"
CIFAR10_TFLITE_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/CIFAR10-quantized.tflite"
CIFAR10_TFLITE_PRUNED_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/CIFAR10-pruned-50-quantized.tflite"

cp $CIFAR10_TFLITE_FLOAT_MODEL_PATH "$CIFAR10_TFLM_PATH/models/cifar10_float.tflite"
cp $CIFAR10_TFLITE_INT8_MODEL_PATH "$CIFAR10_TFLM_PATH/models/cifar10_int8.tflite"
cp $CIFAR10_TFLITE_PRUNED_INT8_MODEL_PATH "$CIFAR10_TFLM_PATH/models/cifar10_pruned_int8.tflite"

xxd -i "$CIFAR10_TFLM_PATH/models/cifar10_float.tflite" > "$CIFAR10_TFLM_PATH/models/generated_cifar10_float_model.h"
xxd -i "$CIFAR10_TFLM_PATH/models/cifar10_int8.tflite" > "$CIFAR10_TFLM_PATH/models/generated_cifar10_int8_model.h"
xxd -i "$CIFAR10_TFLM_PATH/models/cifar10_pruned_int8.tflite" > "$CIFAR10_TFLM_PATH/models/generated_cifar10_pruned_int8_model.h"

make -f tensorflow/lite/micro/tools/make/Makefile test_cifar10_test


# fmnist
FMNIST_TFLITE_FLOAT_MODEL_PATH="$TFLITE_ROOT/tflite/FMNIST.tflite"
FMNIST_TFLITE_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/FMNIST-quantized.tflite"
FMNIST_TFLITE_PRUNED_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/FMNIST-pruned-50-quantized.tflite"

FMNIST_TFLM_PATH="tensorflow/lite/micro/examples/fmnist"
cp $FMNIST_TFLITE_FLOAT_MODEL_PATH "$FMNIST_TFLM_PATH/models/fmnist_float.tflite"
cp $FMNIST_TFLITE_INT8_MODEL_PATH "$FMNIST_TFLM_PATH/models/fmnist_int8.tflite"
cp $FMNIST_TFLITE_PRUNED_INT8_MODEL_PATH "$FMNIST_TFLM_PATH/models/fmnist_pruned_int8.tflite"

xxd -i "$FMNIST_TFLM_PATH/models/fmnist_float.tflite" > "$FMNIST_TFLM_PATH/models/generated_fmnist_float_model.h"
xxd -i "$FMNIST_TFLM_PATH/models/fmnist_int8.tflite" > "$FMNIST_TFLM_PATH/models/generated_fmnist_int8_model.h"
xxd -i "$FMNIST_TFLM_PATH/models/fmnist_pruned_int8.tflite" > "$FMNIST_TFLM_PATH/models/generated_fmnist_pruned_int8_model.h"

make -f tensorflow/lite/micro/tools/make/Makefile test_fmnist_test


# mnist
MNST_TFLITE_FLOAT_MODEL_PATH="$TFLITE_ROOT/tflite/MNIST.tflite"
MNST_TFLITE_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/MNIST-quantized.tflite"
MNST_TFLITE_PRUNED_INT8_MODEL_PATH="$TFLITE_ROOT/tflite/MNIST-pruned-50-quantized.tflite"

MNST_TFLM_PATH="tensorflow/lite/micro/examples/mnist"
cp $MNST_TFLITE_FLOAT_MODEL_PATH "$MNST_TFLM_PATH/models/mnist_float.tflite"
cp $MNST_TFLITE_INT8_MODEL_PATH "$MNST_TFLM_PATH/models/mnist_int8.tflite"
cp $MNST_TFLITE_PRUNED_INT8_MODEL_PATH "$MNST_TFLM_PATH/models/mnist_pruned_int8.tflite"


xxd -i "$MNST_TFLM_PATH/models/mnist_float.tflite" > "$MNST_TFLM_PATH/models/generated_mnist_float_model.h"
xxd -i "$MNST_TFLM_PATH/models/mnist_int8.tflite" > "$MNST_TFLM_PATH/models/generated_mnist_int8_model.h"
xxd -i "$MNST_TFLM_PATH/models/mnist_pruned_int8.tflite" > "$MNST_TFLM_PATH/models/generated_mnist_pruned_int8_model.h"

make -f tensorflow/lite/micro/tools/make/Makefile test_mnist_test
g++ \
    -std=c++14 \
    -fPIC \
    -O3 \
    -DONNX_NAMESPACE=onnx \
    custom_transform.cpp \
    newipucopy.cpp \
    -I ./ \
    -shared \
    -lpopart \
    -o custom_library.so

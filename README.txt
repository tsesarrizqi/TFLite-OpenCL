Build:
bazel build -c opt --cxxopt='--std=c++11' --linkopt='-llog' --linkopt='-lOpenCL' //tensorflow/contrib/lite/java:tensorflowlite --crosstool_top=//external:android/crosstool --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cpu=armeabi-v7a

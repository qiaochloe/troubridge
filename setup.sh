cd ~
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-amd64
chmod +x bazelisk-linux-amd64
./bazelisk-linux-amd64
sudo mv bazelisk-linux-amd64 /usr/local/bin/bazel
cd troubridge
bazel test //tcmalloc/...

bazel build tcmalloc/testing:hello_main
bazel run tcmalloc/testing:hello_main
./bazel-bin/tcmalloc/testing/hello_main

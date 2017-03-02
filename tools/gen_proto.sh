#!/bin/sh

output=${1:-vendor}

echo $output

mkdir -p $output

mkdir -p protobuf/src/google/protobuf
cp $GOPATH/src/github.com/golang/protobuf/ptypes/any/any.proto protobuf/src/google/protobuf/any.proto
cp $GOPATH/src/github.com/golang/protobuf/ptypes/wrappers/wrappers.proto protobuf/src/google/protobuf/wrappers.proto

protoc -I=. -I=./tensorflow -I=./protobuf/src --go_out=plugins=grpc:$output ./tensorflow_serving/apis/*.proto

protoc -I=./tensorflow --go_out=plugins=grpc:$output tensorflow/tensorflow/core/example/*.proto
protoc -I=./tensorflow --go_out=plugins=grpc:$output tensorflow/tensorflow/core/framework/*.proto

protoc -I=./tensorflow -I=./protobuf/src --go_out=plugins=grpc:$output \
	tensorflow/tensorflow/core/protobuf/saver.proto  \
	tensorflow/tensorflow/core/protobuf/meta_graph.proto

rm -r protobuf

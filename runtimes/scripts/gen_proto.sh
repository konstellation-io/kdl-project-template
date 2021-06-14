#!/bin/bash
set -e

protoc -I=krt \
      --python_out=krt/processes/etl \
      --python_out=krt/processes/output \
      krt/public_input.proto
echo "Public interfaces generated"

protoc -I=krt/processes \
      --python_out=krt/processes/etl \
      --python_out=krt/processes/model \
      --python_out=krt/processes/output \
      krt/processes/private.proto
echo "Private interfaces generated"

echo "Done"

#!/bin/bash
set -e

protoc -I=kdl-project-template \
      --python_out=kdl-project-template/processes/etl \
      --python_out=kdl-project-template/processes/output \
      kdl-project-template/public_input.proto
echo "Public interfaces generated"

protoc -I=kdl-project-template/processes \
      --python_out=kdl-project-template/processes/etl \
      --python_out=kdl-project-template/processes/model \
      kdl-project-template/processes/private.proto
echo "Private interfaces generated"

echo "Done"

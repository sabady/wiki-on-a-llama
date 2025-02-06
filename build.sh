#!/bin/bash

podman build --jobs=4 --platform="linux/arm64,linux/amd64" --progress=plain --layers=true --compress --format=docker --manifest "harbor.belong.life/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER}" -f ./Dockerfile 
podman manifest push --all harbor.belong.life/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER} docker://harbor.belong.life/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER}


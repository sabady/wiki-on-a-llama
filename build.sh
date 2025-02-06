#!/bin/bash

podman build --jobs=4 --platform="linux/arm64,linux/amd64" --progress=plain --layers=true --compress --format=docker --manifest "harbor.example.com/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER}" -f ./Dockerfile 
podman manifest push --all harbor.example.com/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER} docker://harbor.example.com/devops/wiki-on-a-llama:${version}_${BUILD_NUMBER}


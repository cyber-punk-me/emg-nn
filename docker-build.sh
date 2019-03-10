#!/bin/bash
docker build -t emg-nn:nvidia docker-nvidia 
docker build -t emg-nn:cpu docker-cpu 

docker tag emg-nn:nvidia kyr7/emg-nn:nvidia
docker tag emg-nn:cpu kyr7/emg-nn:cpu

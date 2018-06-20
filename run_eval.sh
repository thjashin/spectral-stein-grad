#!/bin/bash

for i in $(seq 1234 1243); do
    echo $i;
    echo $1;
    echo $2;
    CUDA_VISIBLE_DEVICES=$2 python -m vae.vae_eval --dir=$1 --seed=$i;
done


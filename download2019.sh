#!/usr/bin/env bash

for url in $(cat urls2019.txt | tr -d '\r')
do
    wget $url --no-check-certificate
done
mkdir /opt/ml/input/data/ICDAR19_MLT
mkdir /opt/ml/input/data/ICDAR19_MLT/images
mkdir /opt/ml/input/data/ICDAR19_MLT/gt

unzip ImagesPart1.zip -d /opt/ml/input/data/ICDAR19_MLT/images
unzip ImagesPart2.zip -d /opt/ml/input/data/ICDAR19_MLT/images
unzip train_gt_t13.zip -d /opt/ml/input/data/ICDAR19_MLT/gt

#!/bin/sh

mkdir -p data
wget https://nlp.cs.washington.edu/ambigqa/data/nqopen.zip -O data/nqopen.zip
unzip -d data data/nqopen.zip
rm data/nqopen.zip


#!/bin/bash
python3 train.py
cd cpp_filter
g++ -std=c++17 main.cpp filter.cpp vectorizer.cpp -o toxfilter
./toxfilter "$@"
cd ..

#!/bin/sh  

python3 src/train.py --fold 0 --model rf  
python3 src/train.py --fold 1 --model rf  
python3 src/train.py --fold 2 --model rf  
python3 src/train.py --fold 3 --model rf  
python3 src/train.py --fold 4 --model rf

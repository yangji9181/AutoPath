#!/bin/bash

for DATASET in $@
do
	case $DATASET in
		yelp)
			DATA_DIR=yelp/
		;;
		imdb)
			DATA_DIR=imdb/
		;;
		dblp)
		;;
	esac

	python3 experiments.py --dataset $DATASET --data_dir $DATA_DIR

done

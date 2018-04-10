#!/bin/bash

for DATASET in $@
do
	case $DATASET in
		yelp)
		;;
		imdb)
			DATA_DIR=imdb/
		;;
		dblp)
		;;
	esac

	python experiments.py --dataset $DATASET --data_dir $DATA_DIR

done

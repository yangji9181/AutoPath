#!/bin/bash

for DATASET in $@
do
	case $DATASET in
		yelp)
		;;
		imdb)
			DATA_DIR=/shared/data/xikunz2/autopath/AutoPath/data/imdb
		;;
		dblp)
		;;
	esac

	python experiments.py --dataset $DATASET --data_dir $DATA_DIR

done

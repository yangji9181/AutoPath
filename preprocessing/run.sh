#!/bin/bash

for DATASET in $@
do
	case $DATASET in
		yelp)
			python extract_data.py --dataset yelp --data_dir ../../yelp_data --pos_pair_num 40000 --neg_sampling_ratio 1
		;;
		imdb)
			python extract_data.py --dataset imdb --data_dir ../../imdb_data/IMDBLens-by-time --pos_pair_num 8000 --neg_sampling_ratio 1
		;;
		dblp)
			python extract_data.py --dataset dblp --data_dir ../../dblp_data --pos_pair_num 8000 --neg_sampling_ratio 1
		;;
	esac

done

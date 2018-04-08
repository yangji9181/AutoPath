# AutoPath
The AutoPath pipeline for similarity modeling on heterogeneous networks with automatic path discovery

## Preprocessing
We first need to preprocess the HIN data. The script provides support for Yelp, IMDb, and DBLP preprocessing functionality. 

First, we change to the preprocessing directory. 
```
cd preprocessing
```
Second, we open the `run.sh` file to set the parameters we want. There are default ones set. 

Third, we run the `run.sh` script followed by **all** the datasets we want to preprocess. For example, 
```
bash run.sh yelp imdb dblp
```
or
```
bash run.sh yelp dblp
```
.




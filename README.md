# AutoPath
The AutoPath pipeline for similarity modeling on heterogeneous networks with automatic path discovery

## Publication
Carl Yang, Mengxiong Liu, Frank He, Jian Peng, Jiawei Han, ``Similarity Modeling on Heterogeneous Networks via Automatic Path Discovery'', submitted to ECML/PKDD 2018, under review.

## Deployment
AutoPath is implemented with TensorFlow and Python2. Please make sure you have the newest version of both of them. If not sure, simply run
```
pip2 install --upgrade pip
pip2 install --upgrade tensorflow
```

## Model Training
Use the following commands to run train the AutoPath model on our default IMDb dataset.
```
cd model
python2 train.py
```
You may also change the parameters as you like in config.py. With our default parameter settings, training the model roughly takes a few minutes on CPU. Note that, you need to remove the tmp folder if you change the neural network structures in order to train a new model.

## Demo
Use the following command to play with the trained AutoPath model.
```
python2 demo.py
```
This demo basically allows you to look for movies with similar genres. Note that exactly finding movies with the exactly the same set of genres is very challenging, because genres can be multiple and ambiguous. Nonetheless, our results show that most movies returned by our model at least share a few genres with the queried movie. To play with the demo, input one or more movies (separated by space) at each time. 

## Example Input
```
0
```
or
```
0 1 2
```

## Example Output
```
Top 10 related movies for <the_van>, ['Comedy', 'Drama']
<michael_collins>, ['Drama', 'Biography', 'Thriller', 'War']
<the_butcher_boy>, ['Comedy', 'Drama']
<his_girl_friday>, ['Comedy', 'Romance', 'Drama']
<the_madness_of_king_george>, ['Comedy', 'Drama', 'Biography', 'History']
<the_cook_the_thief_his_wife_&_her_lover>, ['Crime', 'Comedy', 'Horror', 'Romance', 'Drama']
<tin_men>, ['Comedy', 'Drama']
<the_beverly_hillbillies>, ['Comedy', 'Family']
<an_american_in_paris>, ['Romance', 'Musical']
<sweet_nothing>, ['Drama']
```

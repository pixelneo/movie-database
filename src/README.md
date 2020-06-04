# Usage 

1. Run `pip3 install -r ../requirements.txt`
2. Run `python3 -m spacy download en_core_web_sm` 
3. Download dataset from [Kaggle](https://www.kaggle.com/jrobischon/wikipedia-movie-plots/data#wiki_movie_plots_deduped.csv)
4. Unzip it to folder `../data.nosync`
5. Run `./run.py reproduce METHOD > METHOD.out`. `METHOD` can be either `lda`, `lsa` or `doc2vec`.
Please note that running `lda` takes a lot of time (~ hour).

`./run.py {create|train|inf|reproduce}`

The `reproduce` action runs `create`, `train`, and `inf`.

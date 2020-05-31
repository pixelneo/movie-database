# movie-database
A project for course [Language Technologies in Practice](https://ufal.mff.cuni.cz/courses/npfl128).

## Data sources
- A dump of various movie data (from 2017) from [IMDB](imdb.com): [ftp://ftp.fu-berlin.de/pub/misc/movies/database](ftp://ftp.fu-berlin.de/pub/misc/movies/database) .
It contains about 4.5M movies and tv show episodes (each episode counted as an entry).
For about 600k of them, there is a plot.
- OR a Kaggle dataset [https://www.kaggle.com/jrobischon/wikipedia-movie-plots/data#wiki_movie_plots_deduped.csv](https://www.kaggle.com/jrobischon/wikipedia-movie-plots/data#wiki_movie_plots_deduped.csv) containing about 35k movie plots from wikipedia. They are much longer then those from IMDB.
- OR this dataset [http://www.cs.cmu.edu/~ark/personas/](http://www.cs.cmu.edu/~ark/personas/) - 42k movie plot from wikipedia

## An (optimistic) plan
- [ ] Do topic modelling on movie plots, using multiple methods
    * [x] LDA
    * [x] Doc2Vec
    * [ ] LSA
- [ ] Make a search system: based on title, search for similar movies
    * [ ] TF-IDF or something similar to find the plot of the movie, then use methods from topic modelling to find similar movies.
    * [ ] (maybe) use reviews to cluster the movies - similar movie might have similar reviews
- [ ] TBD

## TODOs
- [x] Remove proper names (eg. john, paul, george, ...)
- [ ] Make documentation on how to run it.
- How to evaluate? 35k plots is not a lot and ~2k test set is very small.

## Results
- LDA + Clustering does not work as expected. There is a big difference between sizes of clusters. And after 'human evaluation' the movies usually are not clustered as expected.
- LDA + `similarities` module from gensim works better than clustering

## Example outputs
`q` is the query, `v` are 5 most similar movies
### LDA
*(These are with ommited query from the search results)*
~~~
q: The Lord of the Rings: The Fellowship of the Ring 2001
v: The Lord of the Rings 1978, The Lord of the Rings: The Return of the King 2003, The Return of the King 1980, The Lord of the Rings: The Two Towers 2002, 23 2016,

q: Casino 1995
v: To Live and Die in L.A. 1985, Goodfellas 1990, Midnight Run 1988, Black Heat 1976, Jackie Brown 1997
~~~

### Doc2Vec
~~~
q: Goodfellas 1990
v: Goodfellas 1990, Street Bandits 1951, Public Enemies 1996, Legend 2015, Fog Over Frisco 1934, Casino 1995

q: Goldfinger 1964
v: Goldfinger 1964, The World Is Not Enough 1999, Casino Royale 2006, Dr. No 1962, A View to a Kill 1985, Skyfall 2012,
~~~

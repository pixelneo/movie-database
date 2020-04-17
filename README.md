# movie-database
A project for course [Language Technologies in Practice](https://ufal.mff.cuni.cz/courses/npfl128).

## Data sources
- A dump of various movie data (from 2017) from [IMDB](imdb.com) [ftp://ftp.fu-berlin.de/pub/misc/movies/database](ftp://ftp.fu-berlin.de/pub/misc/movies/database).
It contains about 4.5M movies and tv show episodes (each episode counted as an entry).
For about 600k of them, there is a plot.

## An (optimistic) plan
- [ ] Do topic modelling on movie plots, using multiple methods
    * [ ] LDA
    * [ ] Doc2Vec + Clustering
    * [ ] LSA + Clustering
- [ ] Make an search system: based on title, search for similar movies
    * [ ] TF-IDF or something similar to find the plot of the movie, then use methods from topic modelling to find similar movies.
    * [ ] (maybe) use reviews to cluster the movies - similar movie might have similar reviews
- [ ] TBD
help:
	echo "help"

download:
	mkdir -p data.nosync
	wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/plot.list.gz -P data.nosync/
	wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/movies.list.gz -P data.nosync/

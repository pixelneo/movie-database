help:
	echo "help"

download:
	mkdir -p data.nosync
	wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/plot.list.gz -P data.nosync/
	wget ftp://ftp.fu-berlin.de/pub/misc/movies/database/frozendata/movies.list.gz -P data.nosync/

reqs:
	pip3 install -r requirements.txt

init: reqs 
	python3 -m spacy download en_core_web_sm

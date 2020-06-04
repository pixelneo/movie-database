# Usage 

1. Run `pip3 install -r ../requirements.txt`
2. Run `python3 -m spacy download en_core_web_sm` 
3. Download dataset from Kaggle
4. Unzip it to folder `../data.nosync`
5. Run `./run.py reproduce`

`./run.py {create|train|inf|reproduce}`

The `reproduce` action runs `create`, `train`, and `inf`.

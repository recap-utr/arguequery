# ReCAP: Argument Graph Retrieval

This program has been used to perform the evaluation for my Bachelor's Thesis.
It provides a retrieval for argumentation graphs.

## System Requirements

- Python 3.7 (Use [pyenv](https://github.com/pyenv/pyenv) for multiple versions)
- [Poetry](https://github.com/sdispater/poetry)

## Installation

### Application

_Please note:_ The `x` in the following command needs to be replaced with a specific version number.

It is important to select a version of Python 3.7 as the active interpreter before installing the packages.
Otherwise, poetry will use the system default Python version (which is most likely Python 2.x).

```shell
pyenv install 3.7.x
pyenv local 3.7.x
```

The dependencies itself can be installed via Poetry.
It also creates a virtual environment to avoid package version errors.

```shell
poetry install
```

Lastly, duplicate the file `config_example.yml` to `config.yml` and adapt the settings to your liking.



### Embeddings

TODO: Paper citations

The following list contains all models used in the paper together with instructions to make them usable for the software.
It is recommended to rename the files to a memorable name and put them in a folder named `data/embeddings`.

- [Google Word2Vec:](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
  - `poetry run python -m recap_agr.cli.convert bytes-text path/to/GoogleNews-vectors-negative300.bin.gz`
  - `poetry run python -m recap_agr.cli.convert model-gensim path/to/GoogleNews-vectors-negative300.txt`
- Custom Doc2Vec: Not yet available.
- [fastText:](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip)
  - Unpack the file.
  - `poetry run python -m recap_agr.cli.convert model-gensim path/to/crawl-300d-2M.vec`
- [GloVe:](http://nlp.stanford.edu/data/glove.840B.300d.zip)
  - Unpack the file.
  - Run `cat path/to/glove.6B.300d.txt | wc -l` to obtain the number of items.
  - Add `#LINES 300` as the first line of the file, e.g. `1000 300` if the output above gave 1000 (recommended to use `vim`).
  - `poetry run python -m recap_agr.cli.convert model-gensim path/to/glove.6B.300d.txt`
- [Infersent:](https://dl.fbaipublicfiles.com/infersent/infersent1.pkl) No modification needed.
- [USE-D:](https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed) Unpack the file.
- [USE-T:](https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed) Unpack the file.




## Usage

It is possible to run the software with

```poetry run python -m recap_agr```

This will start a web server using Flask and the parameters given in `config.yml`.
The terminal will show the complete URL to access the interface.



## Data Folder Contents

The following folders need to be specified:

- `casebase_folder`
- `queries_folder`
- `embeddings_folder`
- `candidates_folder`
- `results_folder`

### Case-Base and Queries

All files need to be present in the AIF- or OVA-format (and thus be `.json` files).


### Embeddings

Only the native `gensim` format is supported.


### Results

No file needs to be put in here.
The exporter will write the results to this folder.
However, the folder needs to be created manually.


### Candidates

For each query, a candidates file with the following content has to be provided so that the evaluation metrics are calculated.

_Please note:_ Candidates and rankings do not need to contain the same filenames.

```json
{
	"candidates": [
		"nodeset6366.json",
		"nodeset6383.json",
		"nodeset6387.json",
		"nodeset6391.json",
		"nodeset6450.json",
		"nodeset6453.json",
		"nodeset6464.json",
		"nodeset6469.json"
	],
	"rankings": {
		"nodeset6366.json": 2,
		"nodeset6383.json": 2,
		"nodeset6387.json": 3,
		"nodeset6391.json": 2,
		"nodeset6450.json": 2,
		"nodeset6453.json": 2,
		"nodeset6464.json": 2,
		"nodeset6469.json": 1
	}
}

```


## Important Information

- [License](LICENSE)
- [Copyright](NOTICE.md)

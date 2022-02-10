# ReCAP: Argument Graph Retrieval

[![DOI](https://zenodo.org/badge/192173055.svg)](https://zenodo.org/badge/latestdoi/192173055)

This program has been used to perform the evaluation for my Bachelor's Thesis.
It provides a retrieval for argumentation graphs.

## System Requirements and Installation

This app is developed using [poetry](https://python-poetry.org).
In addition, we provide a docker-compose configuration for easy execution.
In the following, we will only provide the `docker-compose` commands.

## Server

If you want to perform a case-base retrieval with your own dataset, you only need to start the server:

`docker-compose up server`

It will then listen on port `6789`.

## Optional: Evaluation with Built-In Client

In addition to the retrieval server, we also provide a client to automatically evaluate the retrieval results.
In this case, you can start the whole stack:

`docker-compose up`

The evaluation relies on some data to be available.
Details can be found in the following section.

### Case Base and Queries

All files need to be present in a format that can be parsed by the [arguebuf](https://pypi.org/project/arguebuf/) library.

### Queries and Benchmark Data

For each query with name `NAME.json`, a file `NAME.json` with benchmark ranking has to be created.
This data is used to calculate the evaluation metrics.

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

_Please note:_ `candidates` and `rankings` do not need to contain the same filenames.

## Further Information

- [License](LICENSE)
- [Copyright](NOTICE.md)

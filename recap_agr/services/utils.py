from __future__ import absolute_import, annotations

import json
import logging
import re
import sys
from typing import Any, Dict, Generator, List, Set, Tuple

import nltk
import numpy as np
import tensorflow as tf

logger = logging.getLogger("recap")


def preprocess_text(text: str) -> str:
    # TODO: Check for german texts
    #
    # out = text.lower() if config["lowercase"] else text

    # return out.translate(get_umlauts_map())
    return text


def get_tokens(text: str) -> List[str]:
    """Split a string into a set of unique tokens"""

    text_normalized = preprocess_text(text)
    tokens = nltk.word_tokenize(text_normalized, config["language"])

    if config["ignore_stopwords"]:
        stopwords = nltk.corpus.stopwords.words(config["language"])
        tokens = [word for word in tokens if word not in stopwords]

    if config["stemming"]:
        stemmer = get_stemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def get_stemmer():

    return (
        nltk.stem.snowball.GermanStemmer()
        if config["language"] == "german"
        else nltk.stem.snowball.EnglishStemmer()
    )


def get_umlauts_map():
    """Get a table for converting the German umlauts using the str.translate() func."""

    umlaut_dict = {
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
    }

    return {ord(key): val for key, val in umlaut_dict.items()}


def print_progress(iteration, total, prefix="", suffix="", decimals=1, bar_length=100):
    """Call in a loop to create terminal progress bar"""

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = "█" * filled_length + "—" * (bar_length - filled_length)

    sys.stdout.write("\r%s |%s| %s%s %s" % (prefix, bar, percents, "%", suffix)),

    if iteration + 1 == total:
        sys.stdout.write("\n")

    sys.stdout.flush()


def generate_id(current_id: int = 1001) -> Generator:
    """Generate unique id

    Create new id by calling next()
    """

    while True:
        yield current_id
        current_id += 1

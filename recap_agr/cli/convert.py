from __future__ import absolute_import, annotations

import gzip
import json
import logging
import os
import re
from xml.etree import cElementTree as ET

import click
import gensim

logger = logging.getLogger("recap")
logger.setLevel(logging.INFO)


"""
Module for converting models from different formats to the ones used here.

To remove the first line from gzipped files, run the following command:
    gzip -dc $file | tail +2 | gzip -c > $file_new
"""


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    pass


@main.command()
@click.argument("folder")
def xml_json(folder: str) -> None:
    """Convert XML to JSON.

    FOLDER contains XML files that shall be converted to JSON.
    The folder needs to have the following subdirs:
        en_json
        en_xml
        de_json
        de_xml

    The JSON format is compliant with the AIFdb standard.
    Currently, this function needs an equivalent English version
    of the XML files that are already available in JSON.

    It overwrites the English contents in JSON
    with the German ones from the XML.

    The filenames are the same as for the english files
    """

    logger.info("Converting xml for '{}'".format(folder))

    # Define paths
    de_xml_path: str = os.path.join(folder, "de_xml")
    en_json_path: str = os.path.join(folder, "en_json")
    de_json_path: str = os.path.join(folder, "de_json")

    # Get the german nodes from xml files in specified folder
    # and add them to a list
    file_list: list = []
    for filename in sorted(os.listdir(de_xml_path)):
        if not filename.endswith(".xml"):
            continue

        fullname = os.path.join(de_xml_path, filename)
        tree = ET.parse(fullname)
        node_list = []

        for node in tree.iter(tag="edu"):
            node_list.append(node.text)

        file_list.append(node_list)

    # Open english json files and replace node text with
    # previously fetched german node texts
    for filename in sorted(os.listdir(en_json_path)):
        if not filename.endswith(".json"):
            continue

        fullname_en: str = os.path.join(en_json_path, filename)
        fullname_de: str = os.path.join(de_json_path, filename)

        contents: list = file_list.pop(0)

        with open(fullname_en, "r") as file:
            json_data = json.load(file)
            for item in json_data["nodes"]:
                if len(contents) == 0:
                    break
                item["text"] = contents.pop(0)
        with open(fullname_de, "w") as file:
            json.dump(json_data, file, indent=4)


@main.command()
@click.argument("file")
@click.option(
    "--pattern",
    help="Regex pattern that should be searched for in that file.",
    required=True,
)
def multilingual_monolingual(file: str, pattern: str) -> None:
    """Convert multilingual file to monolingual.

    FILE is the path to a gzipped text file that shall be converted from
    multiple language to a single language (German).

    Patterns for different word embeddings:
        attract_repel: 'de_'
        bivcd: '_de'
        numberbatch: '/c/de/'

    The file will be in the same location and have '_mono' appended.
    """

    logger.info(
        "Converting to monolingual for '{}' with pattern '{}'".format(file, pattern)
    )

    basename_ending = os.path.basename(file)
    basename_mono = os.path.splitext(basename_ending)[0]

    folder = os.path.dirname(file)

    basename_multi = basename_mono + "_mono"
    if basename_mono.find("_multi"):
        basename_multi = basename_mono.replace("_multi", "_mono")

    file_out = os.path.join(folder, basename_multi + ".gz")

    if os.path.isfile(file_out):
        os.remove(file_out)

    with gzip.GzipFile(file_out, "w") as fout:
        with gzip.GzipFile(file, "r") as fin:
            for line in fin:
                line_str = line.decode()
                if re.search(pattern, line_str) is not None:
                    out = re.sub(pattern, "", line_str)
                    fout.write(out.encode())


@main.command()
@click.argument("file")
def bytes_text(file: str) -> None:
    """Convert binary model to text model.

    FILE is the path to a model file that shall be converted to a txt file.
    It has to be compressed by the user.
    Currently only works for word2vec models that were created by e.g. gensim.
    """

    logger.info("Converting bytes to string for '{}'".format(file))

    file_out = file + ".txt"

    if os.path.isfile(file_out):
        os.remove(file_out)

    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)

    word_vectors.save_word2vec_format(file_out)


@main.command()
@click.argument("file")
def model_gensim(file: str) -> None:
    """Convert model to gensim format.

    FILE is the path to a model file that shall be converted to an uncompressed file.

    To get the number of items:
        gzcat emb.txt.gz | wc -l
        echo "LINES 300" | gzip -c > tmp.gz

    To add it to the embedding file:
        cat tmp.gz emb.txt.gz > emb-new.txt.gz
    """

    logger.info("Converting compressed to gensim for '{}'".format(file))

    folderpath, filename_input = os.path.split(file)
    filepath_output = os.path.join(folderpath, filename_input.split(".")[0])

    if os.path.isfile(filepath_output):
        os.remove(filepath_output)

    embedding = gensim.models.KeyedVectors.load_word2vec_format(
        file, binary=True if ".bin" in filename_input else False
    )

    embedding.save(filepath_output + ".model")


if __name__ == "__main__":
    main()

import dataclasses
import os
import sys
import json
import random
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class InputExample:
    guid: int
    sentence: str  # "United States can win if Donald Trump was never elected president ."
    labelseq: str  # "B-ORG I-ORG O O O B-PER I-PER O O O O O"
    golds: Optional[dict] = None  # {"United States": "ORG", "Donald Trump": "PER"}
    nonentity_indices: Optional[dict] = None  # [2, 3, 4, 7, 8, 9, 10, 11]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


# @dataclass
# class DataFrame

def is_separate_line(line):
    return line.startswith('-DOCSTART-') or line == '' or line == '\n' or line == '\t\n'


def get_gold_entities(sentence, labelseq):
    # get `golds` from `sentence` and `labelseq` in InputExample
    gold_entities = {}
    tokens = sentence.rstrip().split(' ')
    labels = labelseq.rstrip().split(' ')
    num_tokens = len(tokens)
    num_labels = len(labels)
    if num_tokens != num_labels:
        sys.exit('Tokens\' number({}) does not comply with labels\' number({}).'.format(num_tokens, num_labels))

    gold_entity = ''
    # for i in range(num_labels):
    for i, label in enumerate(labels):
        # print(label)
        if label[0] == 'B':
            # put last gold entity into result dictionary
            if gold_entity:
                gold_entities[gold_entity.rstrip()] = labels[i - 1][2:]
                gold_entity = ''

            # append current token to the gold entity
            gold_entity += tokens[i] + ' '
            # put current gold entity into result dictionary if last label
            if i == num_labels - 1:
                gold_entities[gold_entity.rstrip()] = label[2:]
        elif label[0] == 'I':
            # always append current token to the gold entity first
            gold_entity += tokens[i] + ' '
            # put current gold entity into result dictionary if last label
            if i == num_labels - 1:
                gold_entities[gold_entity.rstrip()] = label[2:]
        elif label[0] == 'O':
            if gold_entity:
                gold_entities[gold_entity.rstrip()] = labels[i - 1][2:]
                gold_entity = ''
        else:
            print(f'{tokens[i]=}\n{labels[i]=}')
            sys.exit('Expecting B/I/O, instead got {}'.format(label[0]))

    return gold_entities


def get_input_examples(filepath):
    input_examples = []
    with open(filepath, 'r', encoding='utf-8') as filename:
        current_sentence = ''
        current_labelseq = ''
        guid = 0
        # iterate over all lines in file, construct sentence and labels sequence
        for i, line in enumerate(filename):
            if not is_separate_line(line):
                # line: "United B-ORG"
                words = line.rstrip().replace('\t', ' ').split(' ')
                current_sentence += words[0] + ' '
                current_labelseq += words[1] + ' '
            else:
                # come across a separate line
                if not current_sentence:
                    continue
                sentence = current_sentence.rstrip()
                labelseq = current_labelseq.rstrip()
                # print(f'{sentence=}\n{labelseq=}')
                golds = get_gold_entities(sentence, labelseq)
                nonentity_indices = [i for i, x in enumerate(labelseq.split(' ')) if x == "O"]
                input_examples.append(
                    InputExample(
                        guid=guid,
                        sentence=sentence,
                        labelseq=labelseq,
                        golds=golds,
                        nonentity_indices=nonentity_indices
                    )
                )
                current_sentence = ''
                current_labelseq = ''
                guid += 1
    return input_examples


def fill_template(input_example, mask_map=None):
    num_golds = len(input_example.golds)
    filled_prompts = {}
    templates = [
        {
            "pos": "<token_span> is a/an <mask> entity .",
            "neg": "<token_span> is not a named entity ."
        },
        {
            "pos": "::: <token_span> == <mask> :::",
            "neg": "::: <token_span> == none :::"
        },
        {
            "pos": "The entity type of <token_span> is <mask> .",
            "neg": "The entity type of <token_span> is none entity ."
        },
        {
            "pos": "<token_span> belongs to <mask> category .",
            "neg": "<token_span> belongs to none category ."
        },
        {
            "pos": "<token_span> should be tagged as <mask> .",
            "neg": "<token_span> should be tagged as none entity ."
        },
        {
            "pos": "<u1> <u2> <u3> <token_span> <u4> <u5> <u6> <mask> <u7> <u8> <u9> <u10>",
            "neg": "<u1> <u2> <u3> <token_span> <u4> <u5> <u6> <u7> <u8> <mask> <u9> <u10>"
        },
    ]
    template = templates[0]

    # use all non-OTHER entities to fill the positive template
    pos_filled_prompts = []
    for token_span, mask in input_example.golds.items():
        pos_filled_prompts.append(
            template['pos'].replace('<token_span>', token_span).replace('<mask>', mask_map[mask] if mask_map else mask.lower())
        )

    # randomly select twice the number of non-OTHER entities OTHER entities to fill the negative template
    neg_filled_prompts = []
    num_nonentities = len(input_example.nonentity_indices)
    neg_limit = round(num_golds * 1.5)
    if num_nonentities > neg_limit:
        for i in random.sample(input_example.nonentity_indices, neg_limit):
            neg_filled_prompts.append(
                template['neg'].replace('<token_span>', input_example.sentence.split(' ')[i])
            )

    filled_prompts['pos'] = pos_filled_prompts
    filled_prompts['neg'] = neg_filled_prompts
    return filled_prompts


def describe_dataset(dataset):
    total_num_tokens = 0
    total_num_labels = 0
    total_num_golds = 0
    tags = []
    for data in dataset:
        # get all ner tags in given dataset
        for tag in data.golds.values():
            tag = tag.lower()
            if tag not in tags:
                tags.append(tag)
        # tokens and labels might not share the same length, this two variables are for checking purpose
        total_num_tokens += len(data.sentence.split(' '))
        total_num_labels += len(data.labelseq.split(' '))
        total_num_golds += len(data.golds)
    # print(f'{data.sentence=}\n{data.golds=}')
    data_summary = {
        "total_num_sentences": len(dataset),
        "total_num_tokens": total_num_tokens,
        "total_num_labels": total_num_labels,
        # there exists identical entities in a single sentence, which results in the inconsistency between the number of
        # occurrences of "B-*"(3377) and `total_num_golds`(3315). to illustrate, consider following sentence:
        # e.g. "Trump and Obama were never friends because CNN claims that Obama used to talk cheap about Trump."
        "total_num_golds": total_num_golds,
        "total_num_nonentity": total_num_labels - total_num_golds,
        "tags": sorted(tags),
    }
    print("Randomly selected sample data:\n{}".format(random.choice(dataset)))
    print("Dataset summary:\n{}".format(data_summary))


def save_dataset(dataset, path):
    mask_maps = {"LOC": "location", "PER": "person", "ORG": "organization", "MISC": "other"}
    # save dataset to path, in pandas' DataFrame format, comma separated
    processed_data = []
    for data in dataset:
        # if processing conll datasets:
        prompts = fill_template(data, mask_maps)
        # prompts = fill_template(data)
        for pos in prompts['pos']:
            processed_data.append((data.sentence, pos))
        for neg in prompts['neg']:
            processed_data.append((data.sentence, neg))
    df = pd.DataFrame(data=processed_data, columns=['sentence', 'prompt'])
    print(df)
    df.to_csv(path, sep=',', index=False, columns=['sentence', 'prompt'])


if __name__ == '__main__':
    # CoNLL2003
    conll03_train = get_input_examples('./data/conll2003/train_ner.txt')  # max sentence: 114
    conll03_devel = get_input_examples('./data/conll2003/devel_ner.txt')
    conll03_test = get_input_examples('./data/conll2003/test_ner.txt')  # max sentence: 118

    # CoNLL2004
    conll04_train = get_input_examples('./data/ptuningv2/CoNLL04/train.txt')
    conll04_devel = get_input_examples('./data/ptuningv2/CoNLL04/dev.txt')
    conll04_test = get_input_examples('./data/ptuningv2/CoNLL04/test.txt')

    # MIT Movie
    mit_movie_train = get_input_examples('./data/mit-movie/engtrain.bio.txt')
    mit_movie_test = get_input_examples('./data/mit-movie/engtest.bio.txt')
    mit_movie_trivial_train = get_input_examples('./data/mit-movie/trivia10k13train.bio.txt')
    mit_movie_trivial_test = get_input_examples('./data/mit-movie/trivia10k13test.bio.txt')

    # MIT Restaurant
    mit_restaurant_train = get_input_examples('./data/mit-restaurant/train.txt')
    mit_restaurant_test = get_input_examples('./data/mit-restaurant/test.txt')

    save_dataset(conll03_train, './data/tmp/conll03_train.csv')
    save_dataset(conll03_devel, './data/tmp/conll03_devel.csv')
    save_dataset(conll03_test, './data/tmp/conll03_test.csv')

    save_dataset(conll04_train, './data/tmp/conll04_train.csv')
    save_dataset(conll04_devel, './data/tmp/conll04_devel.csv')
    save_dataset(conll04_test, './data/tmp/conll04_test.csv')

    # save_dataset(mit_movie_train, './data/tmp/mit_movie_train.csv')
    # save_dataset(mit_movie_test, './data/tmp/mit_movie_test.csv')
    # save_dataset(mit_movie_trivial_train, './data/tmp/mit_movie_trivial_train.csv')
    # save_dataset(mit_movie_trivial_test, './data/tmp/mit_movie_trivial_test.csv')
    #
    # save_dataset(mit_restaurant_train, './data/tmp/mit_restaurant_train.csv')
    # save_dataset(mit_restaurant_test, './data/tmp/mit_restaurant_test.csv')

    # describe_dataset(conll03_train)
    # describe_dataset(conll03_devel)
    # describe_dataset(conll03_test)

    # describe_dataset(conll04_train)
    # describe_dataset(conll04_devel)
    # describe_dataset(conll04_test)

    # describe_dataset(mit_movie_train)
    # describe_dataset(mit_movie_test)
    # describe_dataset(mit_movie_trivial_train)
    # describe_dataset(mit_movie_test)
    #
    # describe_dataset(mit_restaurant_train)
    # describe_dataset(mit_restaurant_test)

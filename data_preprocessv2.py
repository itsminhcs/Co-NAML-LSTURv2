from config import model_name
import pandas as pd
import swifter
import json
from tqdm import tqdm
from os import path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
import importlib
from transformers import AutoTokenizer, ModernBertModel
import torch

try:
    config = getattr(importlib.import_module("config"), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


# ======================================================
# Parse behaviors (GIỮ NGUYÊN)
# ======================================================
def parse_behaviors(source, target, user2int_path):
    print(f"Parse {source}")

    behaviors = pd.read_table(
        source,
        header=None,
        names=["impression_id", "user", "time", "clicked_news", "impressions"],
    )
    behaviors.clicked_news.fillna(" ", inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1

    pd.DataFrame(user2int.items(), columns=["user", "int"]).to_csv(
        user2int_path, sep="\t", index=False
    )

    for row in behaviors.itertuples():
        behaviors.at[row.Index, "user"] = user2int[row.user]

    for row in tqdm(behaviors.itertuples(), desc="Balancing data"):
        positive = iter([x for x in row.impressions if x.endswith("1")])
        negative = [x for x in row.impressions if x.endswith("0")]
        random.shuffle(negative)
        negative = iter(negative)
        pairs = []
        try:
            while True:
                pair = [next(positive)]
                for _ in range(config.negative_sampling_ratio):
                    pair.append(next(negative))
                pairs.append(pair)
        except StopIteration:
            pass
        behaviors.at[row.Index, "impressions"] = pairs

    behaviors = (
        behaviors.explode("impressions")
        .dropna(subset=["impressions"])
        .reset_index(drop=True)
    )
    behaviors[["candidate_news", "clicked"]] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (
                " ".join([e.split("-")[0] for e in x]),
                " ".join([e.split("-")[1] for e in x]),
            )
        ).tolist()
    )
    behaviors.to_csv(
        target,
        sep="\t",
        index=False,
        columns=["user", "clicked_news", "candidate_news", "clicked"],
    )


# ======================================================
# Parse news – ĐÃ SỬA CHO HiRec
# ======================================================
def parse_news(
    source,
    target,
    topic2id_path,
    subtopic2id_path,
    word2int_path,
    entity2int_path,
    mode,
):
    print(f"Parse {source}")

    news = pd.read_table(
        source,
        header=None,
        usecols=[0, 1, 2, 3, 4, 6, 7],
        quoting=csv.QUOTE_NONE,
        names=[
            "id",
            "category",      # topic
            "subcategory",   # subtopic
            "title",
            "abstract",
            "title_entities",
            "abstract_entities",
        ],
    )

    news.title_entities.fillna("[]", inplace=True)
    news.abstract_entities.fillna("[]", inplace=True)
    news.fillna(" ", inplace=True)

    # --------------------------------------------------
    # TRAIN MODE: create mappings
    # --------------------------------------------------
    if mode == "train":
        topic2id = {"<UNK>": 0}
        subtopic2id = {"<UNK>": 0}

        word2freq = {}
        entity2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in topic2id:
                topic2id[row.category] = len(topic2id)
            if row.subcategory not in subtopic2id:
                subtopic2id[row.subcategory] = len(subtopic2id)

            for w in word_tokenize(row.title.lower()):
                word2freq[w] = word2freq.get(w, 0) + 1
            for w in word_tokenize(row.abstract.lower()):
                word2freq[w] = word2freq.get(w, 0) + 1

            for e in json.loads(row.title_entities):
                entity2freq[e["WikidataId"]] = entity2freq.get(
                    e["WikidataId"], 0
                ) + e["Confidence"]

            for e in json.loads(row.abstract_entities):
                entity2freq[e["WikidataId"]] = entity2freq.get(
                    e["WikidataId"], 0
                ) + e["Confidence"]

        word2int = {
            w: i + 1
            for i, (w, f) in enumerate(word2freq.items())
            if f >= config.word_freq_threshold
        }

        entity2int = {
            e: i + 1
            for i, (e, f) in enumerate(entity2freq.items())
            if f >= config.entity_freq_threshold
        }

        pd.DataFrame(topic2id.items(), columns=["topic", "int"]).to_csv(
            topic2id_path, sep="\t", index=False
        )
        pd.DataFrame(subtopic2id.items(), columns=["subtopic", "int"]).to_csv(
            subtopic2id_path, sep="\t", index=False
        )
        pd.DataFrame(word2int.items(), columns=["word", "int"]).to_csv(
            word2int_path, sep="\t", index=False
        )
        pd.DataFrame(entity2int.items(), columns=["entity", "int"]).to_csv(
            entity2int_path, sep="\t", index=False
        )

    # --------------------------------------------------
    # TEST / DEV MODE: load mappings
    # --------------------------------------------------
    else:
        topic2id = dict(pd.read_table(topic2id_path).values.tolist())
        subtopic2id = dict(pd.read_table(subtopic2id_path).values.tolist())
        word2int = dict(pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

    # --------------------------------------------------
    # Parse each news row
    # --------------------------------------------------
    def parse_row(row):
        new_row = [
            row.id,
            topic2id.get(row.category, 0),
            subtopic2id.get(row.subcategory, 0),
            [0] * config.num_words_title,
            [0] * config.num_words_abstract,
            [0] * config.num_words_title,
            [0] * config.num_words_abstract,
        ]

        local_entity_map = {}
        for e in json.loads(row.title_entities):
            if e["WikidataId"] in entity2int:
                for x in " ".join(e["SurfaceForms"]).lower().split():
                    local_entity_map[x] = entity2int[e["WikidataId"]]

        try:
            for i, w in enumerate(word_tokenize(row.title.lower())):
                if w in word2int:
                    new_row[3][i] = word2int[w]
                    if w in local_entity_map:
                        new_row[5][i] = local_entity_map[w]
        except IndexError:
            pass

        return pd.Series(
            new_row,
            index=[
                "id",
                "topic_id",
                "subtopic_id",
                "title",
                "abstract",
                "title_entities",
                "abstract_entities",
            ],
        )

    parsed_news = news.swifter.apply(parse_row, axis=1)
    parsed_news.to_csv(target, sep="\t", index=False)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    train_dir = "./data/train"
    val_dir = "./data/test"

    parse_behaviors(
        path.join(train_dir, "behaviors.tsv"),
        path.join(train_dir, "behaviors_parsed.tsv"),
        path.join(train_dir, "user2int.tsv"),
    )

    parse_news(
        path.join(train_dir, "news.tsv"),
        path.join(train_dir, "news_parsed.tsv"),
        path.join(train_dir, "topic2id.tsv"),
        path.join(train_dir, "subtopic2id.tsv"),
        path.join(train_dir, "word2int.tsv"),
        path.join(train_dir, "entity2int.tsv"),
        mode="train",
    )

    parse_news(
        path.join(val_dir, "news.tsv"),
        path.join(val_dir, "news_parsed.tsv"),
        path.join(train_dir, "topic2id.tsv"),
        path.join(train_dir, "subtopic2id.tsv"),
        path.join(train_dir, "word2int.tsv"),
        path.join(train_dir, "entity2int.tsv"),
        mode="test",
    )

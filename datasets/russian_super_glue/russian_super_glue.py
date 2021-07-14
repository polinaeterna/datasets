"""
The RussianSuperGLUE benchmark.
"""

import json
import os

import datasets


_RUSSIAN_SUPER_GLUE_CITATION = """
@article{shavrina2020russiansuperglue,
                  title={RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark},
                  author={Shavrina, Tatiana and Fenogenova, Alena and Emelyanov, Anton and Shevelev, Denis and Artemova, Ekaterina and Malykh, Valentin and Mikhailov, Vladislav and Tikhonova, Maria and Chertok, Andrey and Evlampiev, Andrey},
                  journal={arXiv preprint arXiv:2010.15925},
                  year={2020}
                  }
                  
Note that some RussianSuperGLUE datasets has their own citations.
MuSeRC and RuCoS:
(COLING-2020): Read and Reason with MuSeRC and RuCoS: Datasets for MachineReading Comprehension for Russian.
RUSSE:
Panchenko, A., Lopukhina, A., Ustalov, D., Lopukhin, K., Arefyev, N., Leontyev, A., 
Loukachevitch, N.: RUSSE’2018: A Shared Task on Word Sense Induction for the Russian Language. 
In: Computational Linguistics and Intellectual Technologies: Papers from the Annual International Conference “Dialogue”. pp. 547–564. RSUH, Moscow, Russia (2018)
DaNetQA:
DaNetQA: a yes/no Question Answering Dataset for the Russian Language.
"""

_RUSSIAN_SUPER_GLUE_DESCRIPTION = """
RussianSuperGLUE is an advanced Russian general language understanding evaluation benchmark.
It was developed from scratch for the Russian language,
collected and organized analogically to the SuperGLUE methodology (Wang et al., 2019).
"""


class RussianSuperGlueConfig(datasets.BuilderConfig):
    """BuilderConfig for RussianSuperGLUE."""

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for RussianSuperGLUE.

        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:  # TODO: посмотри че там у них в истории
        # 0.0.1: Initial version.
        super(RussianSuperGlueConfig, self).__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class RussianSuperGlue(datasets.GeneratorBasedBuilder):
    """The RussianSuperGLUE benchmark."""

    BUILDER_CONFIGS = [
        RussianSuperGlueConfig(
            name="danetqa",
            description="",
            features=["question", "passage"],
            data_url="https://russiansuperglue.com/tasks/task_info/DaNetQA",
            citation="",
            url="https://russiansuperglue.com/tasks/download/DaNetQA",
        ),
        RussianSuperGlueConfig(
            name="rcb",
            description="",  # TODO
            features=["premise", "hypothesis"],
            label_classes=["entailment", "contradiction", "neutral"],
            data_url="https://russiansuperglue.com/tasks/download/RCB",
            citation="",  # TODO
            url="https://russiansuperglue.com/tasks/task_info/RCB",
        ),
        RussianSuperGlueConfig(
            name="parus",
            description="",
            features=["premise", "choice1", "choice2", "question"],
            # Note that question will only be the X in the statement "What's
            # the X for this?".
            label_classes=["choice1", "choice2"],
            data_url="https://russiansuperglue.com/tasks/download/PARus",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/PARus",
        ),
        RussianSuperGlueConfig(
            name="muserc",
            description="",
            features=["paragraph", "question", "answer"],
            data_url="https://russiansuperglue.com/tasks/download/MuSeRC",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/MuSeRC",
        ),
        RussianSuperGlueConfig(
            name="rucos",  # RECORD
            description="",
            # Note that entities and answers will be a sequences of strings. Query
            # will contain @placeholder as a substring, which represents the word
            # to be substituted in.
            features=["passage", "query", "entities", "answers"],
            data_url="https://russiansuperglue.com/tasks/download/RuCoS",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/RuCoS",
        ),
        RussianSuperGlueConfig(
            name="terra",
            description="",
            features=["premise", "hypothesis"],
            label_classes=["entailment", "not_entailment"],
            data_url="https://russiansuperglue.com/tasks/download/TERRa",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/TERRa",
        ),
        RussianSuperGlueConfig(
            name="russe",
            description="",
            # Note that start1, start2, end1, and end2 will be integers stored as
            # datasets.Value('int32').  # TODO
            features=["word", "sentence1", "sentence2", "start1", "start2", "end1", "end2"],
            data_url="https://russiansuperglue.com/tasks/download/RUSSE",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/RUSSE",
        ),
        RussianSuperGlueConfig(
            name="rwsd",
            description="",
            # Note that span1_index and span2_index will be integers stored as
            # datasets.Value('int32').
            features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
            data_url="https://russiansuperglue.com/tasks/download/RWSD",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/RWSD",
        ),
        # RussianSuperGlueConfig(
        #     name="wsc.fixed",
        #     description=(
        #         _WSC_DESCRIPTION + "\n\nThis version fixes issues where the spans are not actually "
        #         "substrings of the text."
        #     ),
        #     # Note that span1_index and span2_index will be integers stored as
        #     # datasets.Value('int32').
        #     features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
        #     data_url="https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
        #     citation=_WSC_CITATION,
        #     url="https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html",
        # ),
        RussianSuperGlueConfig(
            name="lidirus",
            description="",
            features=["sentence1", "sentence2"],
            label_classes=["entailment", "not_entailment"],
            data_url="https://russiansuperglue.com/tasks/download/LiDiRus",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/LiDiRus",
        )
    ]

    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.name == "rwsd":
            features["span1_index"] = datasets.Value("int32")
            features["span2_index"] = datasets.Value("int32")
        if self.config.name == "russe":
            features["start1"] = datasets.Value("int32")
            features["start2"] = datasets.Value("int32")
            features["end1"] = datasets.Value("int32")
            features["end2"] = datasets.Value("int32")
        if self.config.name == "muserc":
            features["idx"] = dict(
                {
                    "paragraph": datasets.Value("int32"),
                    "question": datasets.Value("int32"),
                    "answer": datasets.Value("int32"),
                }
            )
        elif self.config.name == "rucos":
            features["idx"] = dict(
                {
                    "passage": datasets.Value("int32"),
                    "query": datasets.Value("int32"),
                }
            )
        else:
            features["idx"] = datasets.Value("int32")

        if self.config.name == "rucos":
            features["entities"] = datasets.features.Sequence(datasets.Value("string"))
            # Answers are the subset of entities that are correct.
            features["answers"] = datasets.features.Sequence(datasets.Value("string"))
        else:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)

        return datasets.DatasetInfo(
            description=_RUSSIAN_SUPER_GLUE_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _RUSSIAN_SUPER_GLUE_CITATION,
        )

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        task_name = _get_task_name_from_data_url(self.config.data_url)
        dl_dir = os.path.join(dl_dir, task_name)
        if self.config.name == "lidirus":
            # this is a diagnostic dataset
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(dl_dir, "{}.jsonl".format(task_name)),
                        "split": datasets.Split.TEST,
                    },
                ),
            ]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "val.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_file, split):
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

            if self.config.name == "muserc":
                paragraph = row["passage"]
                for question in paragraph["questions"]:
                    for answer in question["answers"]:
                        label = answer.get("label")
                        key = "%s_%s_%s" % (row["idx"], question["idx"], answer["idx"])
                        yield key, {
                            "paragraph": paragraph["text"],
                            "question": question["question"],
                            "answer": answer["text"],
                            "label": -1 if label is None else _cast_label(bool(label)),
                            "idx": {
                                "paragraph": row["idx"],
                                "question": question["idx"],
                                "answer": answer["idx"],
                            },
                        }

            elif self.config.name == "rucos":
                passage = row["passage"]
                for qa in row["qas"]:
                    yield qa["idx"], {
                        "passage": passage["text"],
                        "query": qa["query"],
                        "entities": _get_record_entities(passage),
                        "answers": _get_record_answers(qa),
                        "idx": {"passage": row["idx"], "query": qa["idx"]},
                    }

            else:
                if self.config.name == "rwsd":
                    row.update(row["target"])
                example = {feature: row[feature] for feature in self.config.features}
                # TODO: check this
                # if self.config.name == "wsc.fixed":
                #     example = _fix_wst(example)

                if "label" in row:
                    if dataset_name == "parus":
                        example["label"] = "choice1" if row["label"] == 0 else "choice2"
                    else:
                        example["label"] = _cast_label(row["label"])
                else:
                    assert split == datasets.Split.TEST, row
                    example["label"] = -1

                yield example["idx"], example


def _cast_label(label):
    """Converts the label into the appropriate string version."""
    if isinstance(label, str):
        return label
    elif isinstance(label, bool):
        return "True" if label else "False"
    elif isinstance(label, int):
        assert label in (0, 1)
        return str(label)
    else:
        raise ValueError("Invalid label format.")


def _get_record_entities(passage):
    """Returns the unique set of entities."""
    text = passage["text"]
    entities = set()
    for entity in passage["entities"]:
        entities.add(text[entity["start"] : entity["end"]])
    return sorted(entities)


def _get_record_answers(qa):
    """Returns the unique set of answers."""
    if "answers" not in qa:
        return []
    answers = set()
    for answer in qa["answers"]:
        answers.add(answer["text"])
    return sorted(answers)


def _get_task_name_from_data_url(data_url):
    return data_url.split("/")[-1].split(".")[0]

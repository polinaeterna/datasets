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
                  
Note that some of the RussianSuperGLUE datasets has their own citations.
MuSeRC and RuCoS:
(COLING-2020): Read and Reason with MuSeRC and RuCoS: Datasets for MachineReading Comprehension for Russian.
RUSSE:
Panchenko, A., Lopukhina, A., Ustalov, D., Lopukhin, K., Arefyev, N., Leontyev, A., 
Loukachevitch, N.: RUSSE’2018: A Shared Task on Word Sense Induction for the Russian Language. 
In: Computational Linguistics and Intellectual Technologies: 
Papers from the Annual International Conference “Dialogue”. pp. 547–564. RSUH, Moscow, Russia (2018)
"""

_RUSSIAN_SUPER_GLUE_DESCRIPTION = """
RussianSuperGLUE is an advanced Russian general language understanding evaluation benchmark.
It was developed from scratch for the Russian language,
collected and organized analogically to the SuperGLUE methodology (Wang et al., 2019).
"""

_DANETQA_DESCRIPTION = """
DaNetQA is a question answering dataset for yes/no questions. These questions are naturally occurring 
-- they are generated in unprompted and unconstrained settings.
Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context.
The text-pair classification setup is similar to existing natural language inference tasks.
By sampling questions from a distribution of information-seeking queries (rather than prompting annotators for text 
pairs), we observe significantly more challenging examples compared to existing NLI datasets.
All text examples were collected in accordance with the methodology for collecting the original dataset 
(SuperGLUE BoolQ). 

Answers to the questions were received with the help of assessors, and texts were also received automatically 
using ODQA systems on Wikipedia. Human assessment was carried out on Yandex.Toloka.
Additionally, to increase number of samples and the distribution of yes/no answers, we added extra data in the same 
format (data were collected from Yandex.Toloka while generating MuSeRC dataset).

SuperGLUE analogue is BoolQ.
"""

_RCB_DESCRIPTION = """
The Russian Commitment Bank is a corpus of naturally occurring discourses whose final sentence contains a 
clause-embedding predicate under an entailment canceling operator (question, modal, negation, antecedent of 
conditional).

All text examples were collected from open news sources and literary magazines, then manually reviewed and supplemented 
by a human assessment on Yandex.Toloka.

SuperGLUE analogue is CommitmentBank.
"""

_PARUS_DESCRIPTION = """
Choice of Plausible Alternatives for Russian language (PARus) evaluation provides researchers with a tool 
for assessing progress in open-domain commonsense causal reasoning. Each question in PARus is composed of a premise 
and two alternatives, where the task is to select the alternative that more plausibly has a causal relation 
with the premise. The correct alternative is randomized so that the expected performance of randomly guessing is 50%. 

All text examples were collected from open news sources and literary magazines, then manually reviewed and 
supplemented by a human assessment on Yandex.Toloka.

SuperGLUE analogue is COPA.
"""

_MUSERC_DESCRIPTION = """
MuSeRC is a reading comprehension challenge in which questions can only be answered by taking into account information 
from multiple sentences. The dataset is the first to study multi-sentence inference at scale, with an open-ended 
set of question types that requires reasoning skills. Task is binary classification of each answer (True/False).

Our challenge dataset contains ∼6k questions for +800 paragraphs across 5 different domains: elementary school texts, 
news, fiction stories, fairy tales, summary of series. First, we have collected all data from open sources 
and automatically preprocessed them, filtered only those paragraphs that corresponding to the following parameters: 
1) paragraph length 2) number of NER entities 3) number of coreference relations. Afterwords we have check 
the correct splitting on sentences and numerate each of them. Next, in Yandex.Toloka we have generated the 
crowdsource task to get from tolkers information: 1) generate questions 2) generate answers 3) check that to solve 
every question man need more than one sentence in the text. 
We exclude any question that can be answered based on a single sentence from a paragraph.
Answers are not written in the full match form in the text.
Answers to the questions are independent from each other. Their number can distinguish.

SuperGLUE analogue is MultiRC.
"""

_RUCOS_DESCRIPTION = """
Russian reading comprehension with Commonsense reasoning (RuCoS) is a large-scale reading comprehension dataset 
which requires commonsense reasoning. RuCoS consists of queries automatically generated from CNN/Daily Mail 
news articles; the answer to each query is a text span from a summarizing passage of the corresponding news. 
The goal of RuCoS is to evaluate a machine`s ability of commonsense reasoning in reading comprehension.

All text examples were collected from open news sources, then automatically filtered with QA systems to 
prevent obvious questions to infiltrate the dataset. The texts were then filtered by IPM frequency of the 
contained words and, finally, manually reviewed.

SuperGLUE analogue is ReCoRD.
"""

_TERRA_DESCRIPTION = """
Textual Entailment Recognition has been proposed recently as a generic task that captures major semantic inference 
needs across many NLP applications, such as Question Answering, Information Retrieval, Information Extraction, 
and Text Summarization. This task requires to recognize, given two text fragments, whether the meaning of one text 
is entailed (can be inferred) from the other text.

All text examples were collected from open news sources and literary magazines, then manually reviewed and 
supplemented by a human assessment on Yandex.Toloka.

SuperGLUE analogue is RTE.
"""

_RUSSE_DESCRIPTION = """
RUSSE is a benchmark for the evaluation of context-sensitive word embeddings.
Depending on its context, an ambiguous word can refer to multiple, potentially unrelated, meanings. 
Mainstream static word embeddings, such as Word2vec and GloVe, are unable to reflect this dynamic semantic nature. 
Contextualised word embeddings are an attempt at addressing this limitation by computing dynamic representations 
for words which can adapt based on context.
RUSSE borrows original data from the Russe project, Word Sense Induction and Disambiguation shared task (2018).

All text examples were collected from Russe original dataset, already collected by Russian Semantic Evaluation 
at ACL SIGSLAV. Human assessment was carried out on Yandex.Toloka.
In version 2, we have manually collected testset in the same format.

SuperGLUE analogue is WiC.
"""

_RWSD_DESCRIPTION = """
A Winograd schema is a pair of sentences that differ in only one or two words and that contain an ambiguity 
that is resolved in opposite ways in the two sentences and requires the use of world knowledge and reasoning 
for its resolution. The strengths of the challenge are that it is clear-cut, in that the answer to each schema 
is a binary choice; vivid, in that it is obvious to non-experts that a program that fails to get the right answers 
clearly has serious gaps in its understanding; and difficult, in that it is far beyond the current state of the art.

All text examples were collected manually translating and adapting original Winograd dataset for Russian. 
Human assessment was carried out on Yandex.Toloka.

SuperGLUE analogue is WSC.
"""

# TODO: check if AXG is analogous.
_LIDIRUS_DESCRIPTION = """
LiDiRus is a diagnostic dataset that covers a large volume of linguistic phenomena, while allowing you to evaluate 
information systems on a simple test of textual entailment recognition.

All text examples manually translated and adapted from English SuperGLUE Diagnostics.

SuperGLUE analogues are AXB and AXG.
"""

_DANETQA_CITATION = """
@article{glushkova2020danetqa,
  title={Danetqa: a yes/no question answering dataset for the russian language},
  author={Glushkova, Taisia and Machnev, Alexey and Fenogenova, Alena and Shavrina, Tatiana and Artemova, 
  Ekaterina and Ignatov, Dmitry I},
  journal={arXiv preprint arXiv:2010.02605},
  year={2020}
}
"""

_MUSERC_CITATION = """
@inproceedings{fenogenova2020read,
  title={Read and Reason with MuSeRC and RuCoS: Datasets for Machine Reading Comprehension for Russian},
  author={Fenogenova, Alena and Mikhailov, Vladislav and Shevelev, Denis},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6481--6497},
  year={2020}
}
"""

_RUSSE_CITATION = """
@article{panchenko2018russe,
  title={RUSSE'2018: a shared task on word sense induction for the Russian language},
  author={Panchenko, Alexander and Lopukhina, Anastasiya and Ustalov, Dmitry and Lopukhin, Konstantin and Arefyev, 
  Nikolay and Leontyev, Alexey and Loukachevitch, Natalia},
  journal={arXiv preprint arXiv:1803.05795},
  year={2018}
}
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
            description=_DANETQA_DESCRIPTION,
            features=["question", "passage"],
            data_url="https://russiansuperglue.com/tasks/task_info/DaNetQA",
            citation=_DANETQA_CITATION,
            url="https://russiansuperglue.com/tasks/download/DaNetQA",
        ),
        RussianSuperGlueConfig(
            name="rcb",
            description=_RCB_DESCRIPTION,
            features=["premise", "hypothesis"],
            label_classes=["entailment", "contradiction", "neutral"],
            data_url="https://russiansuperglue.com/tasks/download/RCB",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/RCB",
        ),
        RussianSuperGlueConfig(
            name="parus",
            description=_PARUS_DESCRIPTION,
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
            description=_MUSERC_DESCRIPTION,
            features=["paragraph", "question", "answer"],
            data_url="https://russiansuperglue.com/tasks/download/MuSeRC",
            citation=_MUSERC_CITATION,
            url="https://russiansuperglue.com/tasks/task_info/MuSeRC",
        ),
        RussianSuperGlueConfig(
            name="rucos",
            description=_RUCOS_DESCRIPTION,
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
            description=_TERRA_DESCRIPTION,
            features=["premise", "hypothesis"],
            label_classes=["entailment", "not_entailment"],
            data_url="https://russiansuperglue.com/tasks/download/TERRa",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/TERRa",
        ),
        RussianSuperGlueConfig(
            name="russe",
            description=_RUSSE_DESCRIPTION,
            # Note that start1, start2, end1, and end2 will be integers stored as
            # datasets.Value('int32').  # TODO
            features=["word", "sentence1", "sentence2", "start1", "start2", "end1", "end2"],
            data_url="https://russiansuperglue.com/tasks/download/RUSSE",
            citation=_RUSSE_CITATION,
            url="https://russiansuperglue.com/tasks/task_info/RUSSE",
        ),
        RussianSuperGlueConfig(
            name="rwsd",
            description=_RWSD_DESCRIPTION,
            # Note that span1_index and span2_index will be integers stored as
            # datasets.Value('int32').
            features=["text", "span1_index", "span2_index", "span1_text", "span2_text"],
            data_url="https://russiansuperglue.com/tasks/download/RWSD",
            citation="",
            url="https://russiansuperglue.com/tasks/task_info/RWSD",
        ),
        # TODO: check this
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
            description=_LIDIRUS_DESCRIPTION,
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

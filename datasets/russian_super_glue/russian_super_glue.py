"""
The RussianSuperGLUE benchmark.
"""

import json
import os

import datasets


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
            name="rucos",
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
        pass

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, data_file, split):
        pass

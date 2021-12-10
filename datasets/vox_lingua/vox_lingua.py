# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""
import glob
import json
import os

import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}
"""

# TODO: Add description of the dataset here
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_DL_URL = "http://bark.phon.ioc.ee/voxlingua107/{name}.zip"

# _LANG_URL = "http://bark.phon.ioc.ee/voxlingua107/zip_urls.txt"


# TODO: what's the best way to provide language codes (there are 107 of them)
with open(os.path.join(os.getcwd(), "languages.json")) as f:
    _LANGUAGES = json.load(f)


class VoxLinguaConfig(datasets.BuilderConfig):
    """BuilderConfig for VoxLinguaDataset."""

    def __init__(self, name, **kwargs):
        super(VoxLinguaConfig, self).__init__(**kwargs)
        self.name = name


class VoxLinguaDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [VoxLinguaConfig(name=lang) for lang in _LANGUAGES]

    def _info(self):
        features = datasets.Features(
            {
                "file": datasets.Value("string"),
                "audio": datasets.features.Audio(sampling_rate=16_000),  # TODO: check sampling rate
                "language": datasets.ClassLabel(names=_LANGUAGES),  # TODO: labels or string?
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = {"train": _DL_URL.format(name=self.config.name), "dev": _DL_URL.format(name="dev")}
        archive_paths = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archive_path": archive_paths["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archive_path": archive_paths["dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, archive_path, split):
        all_paths = glob.glob(os.path.join(archive_path, "**", "*.wav"))
        for path in all_paths:
            lang = path.split("/")[-2] if split == "dev" else self.config.name

            yield path, {
                "file": path,
                "audio": path,
                # it's always the same withing a configuration (except for the "dev" one),
                # just to be convenient for further datasets concats
                "language": lang,
            }

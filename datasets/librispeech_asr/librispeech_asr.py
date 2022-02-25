# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
"""Librispeech automatic speech recognition dataset."""
import glob
import os

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_CITATION = """\
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz,
prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read
audiobooks from the LibriVox project, and has been carefully segmented and aligned.87

Note that in order to limit the required storage for preparing this dataset, the audio
is stored in the .flac format and is not converted to a float32 array. To convert, the audio
file to a float32 array, please make use of the `.map()` function as follows:


```python
import soundfile as sf

def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch

dataset = dataset.map(map_to_array, remove_columns=["file"])
```
"""

_URL = "http://www.openslr.org/12"
_DL_URL = "http://www.openslr.org/resources/12/"

_DL_URLS = {
    "clean": {
        "dev": _DL_URL + "dev-clean.tar.gz",
        "test": _DL_URL + "test-clean.tar.gz",
        "train.100": _DL_URL + "train-clean-100.tar.gz",
        "train.360": _DL_URL + "train-clean-360.tar.gz",
    },
    "other": {
        "test": _DL_URL + "test-other.tar.gz",
        "dev": _DL_URL + "dev-other.tar.gz",
        "train.500": _DL_URL + "train-other-500.tar.gz",
    },
}


class LibrispeechASRConfig(datasets.BuilderConfig):
    """BuilderConfig for LibriSpeechASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(LibrispeechASRConfig, self).__init__(version=datasets.Version("2.1.0", ""), **kwargs)


class LibrispeechASR(datasets.GeneratorBasedBuilder):
    """Librispeech dataset."""

    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIGS = [
        LibrispeechASRConfig(name="clean", description="'Clean' speech."),
        LibrispeechASRConfig(name="other", description="'Other', more challenging, speech."),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "speaker_id": datasets.Value("int64"),
                    "chapter_id": datasets.Value("int64"),
                    "id": datasets.Value("string"),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_URL,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(audio_file_path_column="file", transcription_column="text")],
        )

    def _split_generators(self, dl_manager):
        streaming = dl_manager.is_streaming
        archive_path = dl_manager.download(_DL_URLS[self.config.name])
        if streaming:
            # Here we use iter_archive in streaming mode because dl_manager.download_and_extract
            # doesn't work to stream TAR archives (we have to stream the files in the archive one by one).
            #
            # The iter_archive method returns an iterable of (path_within_archive, file_obj) for every
            # file in the TAR archive.
            #
            archive_iterator = dl_manager.iter_archive(archive_path)
            extracted_dir = None
        else:
            extracted_dir = dl_manager.extract(archive_path)
            archive_iterator = None

        if self.config.name == "clean":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.100", gen_kwargs={
                        "archive_iterator": dl_manager.iter_archive(archive_path["train.100"]),
                        "extracted_path": extracted_dir["train.100"] if extracted_dir else None,
                        "streaming": streaming,
                    }
                ),
                datasets.SplitGenerator(
                    name="train.360", gen_kwargs={
                        "archive_iterator": dl_manager.iter_archive(archive_path["train.360"]),
                        "extracted_path": extracted_dir["train.360"] if extracted_dir else None,
                        "streaming": streaming,
                    }
                ),
            ]

        elif self.config.name == "other":
            train_splits = [
                datasets.SplitGenerator(
                    name="train.500", gen_kwargs={
                        "archive_interator": dl_manager.iter_archive(archive_path["train.500"]),
                        "extracted_path": extracted_dir["train.500"] if extracted_dir else None,
                        "streaming": streaming,
                    }
                ),
            ]

        return train_splits + [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "archive_iterator": dl_manager.iter_archive(archive_path["dev"]),
                    "extracted_path": extracted_dir["dev"] if extracted_dir else None,
                    "streaming": streaming,
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "archive_iterator": dl_manager.iter_archive(archive_path["test"]),
                    "extracted_path": extracted_dir["test"] if extracted_dir else None,
                    "streaming": streaming,
                }
            ),
        ]

    def _generate_examples(self, archive_iterator, archive_path, streaming):
        if streaming:
            yield from self._generate_examples_streaming(archive_iterator)
        else:
            yield from self._generate_examples_non_streaming(archive_path)

    def _generate_examples_non_streaming(self, extracted_path):
        """Generate examples from a LibriSpeech archive_path."""
        transcripts_glob = os.path.join(extracted_path, "LibriSpeech", "*/*/*/*.txt")
        key = 0
        for transcript_path in sorted(glob.glob(transcripts_glob)):
            transcript_dir_path = os.path.dirname(transcript_path)
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    id_, transcript = line.split(" ", 1)
                    audio_file = f"{id_}.flac"
                    speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                    yield key, {
                        "id": id_,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "path": os.path.join(transcript_dir_path, audio_file),
                        "audio": os.path.join(transcript_dir_path, audio_file),
                        "text": transcript,
                    }

    def _generate_examples_streaming(self, archive_iterator):
        """Generate examples from a LibriSpeech archive_path."""
        key = 0
        audio_data = {}
        transcripts = []
        for path, f in archive_iterator:
            if path.endswith(".flac"):
                id_ = path.split("/")[-1][: -len(".flac")]
                audio_data[id_] = f.read()
            elif path.endswith(".trans.txt"):
                for line in f:
                    if line:
                        line = line.decode("utf-8").strip()
                        id_, transcript = line.split(" ", 1)
                        audio_file = f"{id_}.flac"
                        speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                        transcripts.append(
                            {
                                "id": id_,
                                "speaker_id": speaker_id,
                                "chapter_id": chapter_id,
                                "path": audio_file,
                                "text": transcript,
                            }
                        )
            if audio_data and len(audio_data) == len(transcripts):
                for transcript in transcripts:
                    audio = {"path": transcript["path"], "bytes": audio_data[transcript["id"]]}
                    yield key, {"audio": audio, "path": None, **transcript}
                    key += 1
                audio_data = {}
                transcripts = []

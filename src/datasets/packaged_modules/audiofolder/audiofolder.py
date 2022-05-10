import collections
import itertools
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pyarrow.compute as pc
import pyarrow.json as paj

import datasets
from datasets.tasks import AutomaticSpeechRecognition


logger = datasets.utils.logging.get_logger(__name__)


if datasets.config.PYARROW_VERSION.major >= 7:

    def pa_table_to_pylist(table):
        return table.to_pylist()

else:

    def pa_table_to_pylist(table):
        keys = table.column_names
        values = table.to_pydict().values()
        return [{k: v for k, v in zip(keys, row_values)} for row_values in zip(*values)]


def count_path_segments(path):
    cnt = 0
    while True:
        parts = os.path.split(path)
        if parts[0] == path:
            break
        elif parts[1] == path:
            break
        else:
            path = parts[0]
            cnt += 1
    return cnt


@dataclass
class AudioFolderConfig(datasets.BuilderConfig):
    """BuilderConfig for AudioFolder."""

    features: Optional[datasets.Features] = None
    drop_labels: bool = True  # different from ImageFolder as we don't want to infer labels by default
    drop_metadata: bool = False


class AudioFolder(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = AudioFolderConfig

    AUDIO_EXTENSIONS: List[str] = []  # definition at the bottom of the script
    METADATA_FILENAME: str = "metadata.jsonl"

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")

        # Do an early pass if:
        # * `features` are not specified, to infer the class labels
        # * `drop_metadata` is False, to find the metadata files
        do_analyze = (self.config.features is None and not self.config.drop_labels) or not self.config.drop_metadata
        if do_analyze:
            labels = set()
            metadata_files = collections.defaultdict(list)

            def analyze(files_or_archives, downloaded_files_or_dirs, split):
                if len(downloaded_files_or_dirs) == 0:
                    return
                # The files are separated from the archives at this point, so check the first sample
                # to see if it's a file or a directory and iterate accordingly
                if os.path.isfile(downloaded_files_or_dirs[0]):
                    original_files, downloaded_files = files_or_archives, downloaded_files_or_dirs
                    for original_file, downloaded_file in zip(original_files, downloaded_files):
                        original_file, downloaded_file = str(original_file), str(downloaded_file)
                        _, original_file_ext = os.path.splitext(original_file)
                        if original_file_ext.lower() in self.AUDIO_EXTENSIONS:
                            labels.add(os.path.basename(os.path.dirname(original_file)))
                        elif os.path.basename(original_file) == self.METADATA_FILENAME:
                            metadata_files[split].append((original_file, downloaded_file))
                        else:
                            original_file_name = os.path.basename(original_file)
                            logger.debug(
                                f"The file '{original_file_name}' was ignored: it is not an audio, and is not {self.METADATA_FILENAME} either."
                            )
                else:
                    archives, downloaded_dirs = files_or_archives, downloaded_files_or_dirs
                    for archive, downloaded_dir in zip(archives, downloaded_dirs):
                        archive, downloaded_dir = str(archive), str(downloaded_dir)
                        for downloaded_dir_file in dl_manager.iter_files(downloaded_dir):
                            _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                            if downloaded_dir_file_ext in self.AUDIO_EXTENSIONS:
                                labels.add(os.path.basename(os.path.dirname(downloaded_dir_file)))
                            elif os.path.basename(downloaded_dir_file) == self.METADATA_FILENAME:
                                metadata_files[split].append((None, downloaded_dir_file))
                            else:
                                archive_file_name = os.path.basename(archive)
                                original_file_name = os.path.basename(original_file)
                                logger.debug(
                                    f"The file '{original_file_name}' from the archive '{archive_file_name}' was ignored: it is not an audio, and is not {self.METADATA_FILENAME} either."
                                )

            if not self.config.drop_labels:
                logger.info("Inferring labels from data files...")
            if not self.config.drop_metadata:
                logger.info("Analyzing metadata files...")

        data_files = self.config.data_files
        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            files, archives = self._split_files_and_archives(files)
            downloaded_files = dl_manager.download(files)
            downloaded_dirs = dl_manager.download_and_extract(archives)
            if do_analyze:
                analyze(files, downloaded_files, split_name)
                analyze(archives, downloaded_dirs, split_name)
            splits.append(
                datasets.SplitGenerator(
                    name=split_name,
                    gen_kwargs={
                        "files": [(file, downloaded_file) for file, downloaded_file in zip(files, downloaded_files)]
                        + [(None, dl_manager.iter_files(downloaded_dir)) for downloaded_dir in downloaded_dirs],
                        "metadata_files": metadata_files if not self.config.drop_metadata else None,
                        "split_name": split_name,
                    },
                )
            )

        if not self.config.drop_metadata and metadata_files:
            # Verify that:
            # * all metadata files have the same set of features
            # * the `file_name` key is one of the metadata keys and is of type string
            features_per_metadata_file: List[Tuple[str, datasets.Features]] = []
            for _, downloaded_metadata_file in itertools.chain.from_iterable(metadata_files.values()):
                with open(downloaded_metadata_file, "rb") as f:
                    pa_metadata_table = paj.read_json(f)
                features_per_metadata_file.append(
                    (downloaded_metadata_file, datasets.Features.from_arrow_schema(pa_metadata_table.schema))
                )
            for downloaded_metadata_file, metadata_features in features_per_metadata_file:
                if metadata_features != features_per_metadata_file[0][1]:
                    raise ValueError(
                        f"Metadata files {downloaded_metadata_file} and {features_per_metadata_file[0][0]} have different features: {features_per_metadata_file[0]} != {metadata_features}"
                    )
            metadata_features = features_per_metadata_file[0][1]
            if "file_name" not in metadata_features:
                raise ValueError("`file_name` must be present as dictionary key in metadata files")
            if metadata_features["file_name"] != datasets.Value("string"):
                raise ValueError("`file_name` key must be a string")
            del metadata_features["file_name"]
        else:
            metadata_features = None

        # Normally, we would do this in _info, but we need to know the labels and/or metadata
        # before building the features
        if self.config.features is None:
            if not self.config.drop_labels:
                # audio classification is not a canonical task, but we keep inferring labels from directories names
                # to be aligned with ImageFolder
                self.info.features = datasets.Features(
                    {"audio": datasets.Audio(), "label": datasets.ClassLabel(names=sorted(labels))}
                )
            else:
                self.info.features = datasets.Features({"audio": datasets.Audio()})

            if not self.config.drop_metadata and metadata_files:
                # Verify that there are no duplicated keys when compared to the existing features ("audio", optionally "label")
                duplicated_keys = set(self.info.features) & set(metadata_features)
                if duplicated_keys:
                    raise ValueError(
                        f"Metadata feature keys {list(duplicated_keys)} are already present as the audio features"
                    )
                self.info.features.update(metadata_features)

        return splits

    def _split_files_and_archives(self, data_files):
        files, archives = [], []
        for data_file in data_files:
            _, data_file_ext = os.path.splitext(data_file)
            if data_file_ext.lower() in self.AUDIO_EXTENSIONS:
                files.append(data_file)
            elif os.path.basename(data_file) == self.METADATA_FILENAME:
                files.append(data_file)
            else:
                archives.append(data_file)
        return files, archives

    def _generate_examples(self, files, metadata_files, split_name):
        if not self.config.drop_metadata and metadata_files:
            split_metadata_files = metadata_files.get(split_name, [])
            non_metadata_keys = ["audio", "label"] if not self.config.drop_labels else ["audio"]
            audio_empty_metadata = {k: None for k in self.info.features if k not in non_metadata_keys}

            last_checked_dir = None
            metadata_dir = None
            metadata_dict = None

            file_idx = 0
            for original_file, downloaded_file_or_dir in files:
                if original_file is not None:
                    _, original_file_ext = os.path.splitext(original_file)
                    if original_file_ext.lower() in self.AUDIO_EXTENSIONS:
                        # If the file is an audio, and we've just entered a new directory,
                        # find the nereast metadata file (by counting path segments) for the directory
                        current_dir = os.path.dirname(original_file)
                        if last_checked_dir is None or last_checked_dir != current_dir:
                            last_checked_dir = current_dir
                            metadata_dir = None
                            metadata_dict = None
                            metadata_file_candidates = [
                                (
                                    os.path.relpath(original_file, os.path.dirname(metadata_file)),
                                    metadata_file,
                                    downloaded_metadata_file,
                                )
                                for metadata_file, downloaded_metadata_file in split_metadata_files
                                if metadata_file is not None  # ignore metadata_files that are inside archives
                                and not os.path.relpath(original_file, os.path.dirname(metadata_file)).startswith("..")
                            ]
                            if metadata_file_candidates:
                                _, metadata_file, downloaded_metadata_file = min(
                                    metadata_file_candidates, key=lambda x: count_path_segments(x[0])
                                )
                                with open(downloaded_metadata_file, "rb") as f:
                                    pa_metadata_table = paj.read_json(f)
                                pa_file_name_array = pa_metadata_table["file_name"]
                                pa_file_name_array = pc.replace_substring(
                                    pa_file_name_array, pattern="\\", replacement="/"
                                )
                                pa_metadata_table = pa_metadata_table.drop(["file_name"])
                                metadata_dir = os.path.dirname(metadata_file)
                                metadata_dict = {
                                    file_name: audio_metadata
                                    for file_name, audio_metadata in zip(
                                        pa_file_name_array.to_pylist(), pa_table_to_pylist(pa_metadata_table)
                                    )
                                }
                        if metadata_dir is not None:
                            file_relpath = os.path.relpath(original_file, metadata_dir)
                            file_relpath = file_relpath.replace("\\", "/")
                            audio_metadata = metadata_dict.get(file_relpath, audio_empty_metadata)
                        else:
                            audio_metadata = audio_empty_metadata
                        if self.config.drop_labels:
                            yield file_idx, {
                                "audio": downloaded_file_or_dir,
                                **audio_metadata,
                            }
                        else:
                            yield file_idx, {
                                "audio": downloaded_file_or_dir,
                                "label": os.path.basename(os.path.dirname(original_file)),
                                **audio_metadata,
                            }
                        file_idx += 1
                else:
                    for downloaded_dir_file in downloaded_file_or_dir:
                        _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                        if downloaded_dir_file_ext.lower() in self.AUDIO_EXTENSIONS:
                            current_dir = os.path.dirname(downloaded_dir_file)
                            if last_checked_dir is None or last_checked_dir != current_dir:
                                last_checked_dir = current_dir
                                metadata_dir = None
                                metadata_dict = None
                                metadata_file_candidates = [
                                    (
                                        os.path.relpath(
                                            downloaded_dir_file, os.path.dirname(downloaded_metadata_file)
                                        ),
                                        metadata_file,
                                        downloaded_metadata_file,
                                    )
                                    for metadata_file, downloaded_metadata_file in split_metadata_files
                                    if metadata_file is None  # ignore metadata_files that are not inside archives
                                    and not os.path.relpath(
                                        downloaded_dir_file, os.path.dirname(downloaded_metadata_file)
                                    ).startswith("..")
                                ]
                                if metadata_file_candidates:
                                    _, metadata_file, downloaded_metadata_file = min(
                                        metadata_file_candidates, key=lambda x: count_path_segments(x[0])
                                    )
                                    with open(downloaded_metadata_file, "rb") as f:
                                        pa_metadata_table = paj.read_json(f)
                                    pa_file_name_array = pa_metadata_table["file_name"]
                                    pa_file_name_array = pc.replace_substring(
                                        pa_file_name_array, pattern="\\", replacement="/"
                                    )
                                    pa_metadata_table = pa_metadata_table.drop(["file_name"])
                                    metadata_dir = os.path.dirname(downloaded_metadata_file)
                                    metadata_dict = {
                                        file_name: audio_metadata
                                        for file_name, audio_metadata in zip(
                                            pa_file_name_array.to_pylist(), pa_table_to_pylist(pa_metadata_table)
                                        )
                                    }
                            if metadata_dir is not None:
                                downloaded_dir_file_relpath = os.path.relpath(downloaded_dir_file, metadata_dir)
                                downloaded_dir_file_relpath = downloaded_dir_file_relpath.replace("\\", "/")
                                audio_metadata = metadata_dict.get(downloaded_dir_file_relpath, audio_empty_metadata)
                            else:
                                audio_metadata = audio_empty_metadata
                            if self.config.drop_labels:
                                yield file_idx, {
                                    "audio": downloaded_dir_file,
                                    **audio_metadata,
                                }
                            else:
                                yield file_idx, {
                                    "audio": downloaded_dir_file,
                                    "label": os.path.basename(os.path.dirname(downloaded_dir_file)),
                                    **audio_metadata,
                                }
                            file_idx += 1
        else:
            file_idx = 0
            for original_file, downloaded_file_or_dir in files:
                if original_file is not None:
                    _, original_file_ext = os.path.splitext(original_file)
                    if original_file_ext.lower() in self.AUDIO_EXTENSIONS:
                        if self.config.drop_labels:
                            yield file_idx, {
                                "audio": downloaded_file_or_dir,
                            }
                        else:
                            yield file_idx, {
                                "audio": downloaded_file_or_dir,
                                "label": os.path.basename(os.path.dirname(original_file)),
                            }
                        file_idx += 1
                else:
                    for downloaded_dir_file in downloaded_file_or_dir:
                        _, downloaded_dir_file_ext = os.path.splitext(downloaded_dir_file)
                        if downloaded_dir_file_ext.lower() in self.AUDIO_EXTENSIONS:
                            if self.config.drop_labels:
                                yield file_idx, {
                                    "audio": downloaded_dir_file,
                                }
                            else:
                                yield file_idx, {
                                    "audio": downloaded_dir_file,
                                    "label": os.path.basename(os.path.dirname(downloaded_dir_file)),
                                }
                            file_idx += 1


AudioFolder.AUDIO_EXTENSIONS = [
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".htk",
    ".svx",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".mp3",
    ".ogg",
    ".opus",
    ".paf",
    ".pvf",
    ".raw",
    ".rf64",
    ".sd2",
    ".sds",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
    ".wve",
    ".xi",
]

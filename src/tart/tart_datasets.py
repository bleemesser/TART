import os
import sys
from io import BytesIO
from time import sleep
import requests
from PIL import Image
import hashlib
from tqdm import tqdm
from datasets import concatenate_datasets, load_dataset, Dataset

sys.path.append(f"{os.path.dirname(os.getcwd())}/")


from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List


class TartDataset(ABC):
    _domain: str
    _hf_dataset_identifier: str
    _input_key: str

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        self.total_train_samples = total_train_samples
        self.k_range = k_range
        self.seed = seed
        self.cache_dir = cache_dir
        self.data_dir_path = data_dir_path
        self._max_eval_samples = max_eval_samples
        self._pos_class = pos_class
        self._neg_class = neg_class

    @abstractmethod
    def prepare_train_test_split(self):
        pass

    @property
    def pos_class(self):
        return self._pos_class

    @property
    def neg_class(self):
        return self._neg_class

    @property
    def domain(self):
        return self._domain

    @property
    def data_dir(self):
        return self.data_dir_path

    @property
    def input_key(self):
        return self._input_key

    @property
    def hf_dataset_identifier(self):
        return self._hf_dataset_identifier

    @property
    def get_train_df(self):
        return self.train_df

    @property
    def get_test_df(self):
        return self.test_df

    @property
    def max_eval_samples(self):
        return self._max_eval_samples

    @max_eval_samples.setter
    def max_eval_samples(self, value):
        self._max_eval_samples = value

    @property
    def get_dataset(self):
        if self.max_eval_samples:
            # get the label with least number of samples, use this to compute max_eval_samples
            min_label_cnt = self.test_df["label"].value_counts()[
                self.test_df["label"].value_counts().idxmin()
            ]
            self.max_eval_samples = min(self.max_eval_samples, 2 * min_label_cnt)

            test_df_samples = (
                self.test_df.groupby("label")
                .apply(
                    lambda x: x.sample(
                        int(self.max_eval_samples / 2), random_state=self.seed
                    )
                )
                .reset_index(drop=True)
            )
            if len(self.input_key) > 1:
                self.X_test = list(
                    zip(
                        test_df_samples[self.input_key[0]].tolist(),
                        test_df_samples[self.input_key[1]].tolist(),
                    )  # TODO: generalize to more than 2 modalities
                )
                self.y_test = self.test_df["label"].tolist()
            else:
                self.X_test, self.y_test = (
                    test_df_samples[self.input_key].tolist(),
                    test_df_samples["label"].tolist(),
                )
        return self.X_train, self.y_train, self.X_test, self.y_test

    def _prep_train_split(
        self, total_train_samples: int, seed: int, k_range: List[int] = None
    ):
        # sample a class balanced set of train samples from train_df

        min_label_cnt = self.train_df["label"].value_counts()[
            self.test_df["label"].value_counts().idxmin()
        ]

        assert (
            k_range[-1] / 2 <= min_label_cnt
        ), f"k_range should be less than {2 * min_label_cnt}"

        total_train_samples = min(total_train_samples, 2 * min_label_cnt)

        train_df_samples = (
            self.train_df.groupby("label")
            .apply(lambda x: x.sample(int(total_train_samples / 2), random_state=seed))
            .reset_index(drop=True)
        )

        samples = []
        total_sampled = 0
        for k in k_range:
            curr_k = k - total_sampled
            total_sampled = k
            # sample k samples from each class of train_df_samples, and remove the sampled samples from train_df_samples
            k_samples = train_df_samples.groupby("label").apply(
                lambda x: x.sample(int(curr_k / 2), random_state=seed)
            )
            # get second level index
            k_samples = k_samples.droplevel(0)
            train_df_samples = train_df_samples.drop(k_samples.index)
            samples.append(k_samples)

        # concatenate all the samples as a final df
        final_df = pd.concat(samples)

        return final_df

    def _save(self):
        dataset_name = self.hf_dataset_identifier
        if not os.path.exists(os.path.join(self.data_dir, dataset_name)):
            os.makedirs(os.path.join(self.data_dir, dataset_name))

        self.train_df.to_csv(
            os.path.join(self.data_dir, f"{dataset_name}/{dataset_name}_train.csv"),
            index=False,
        )
        self.test_df.to_csv(
            os.path.join(self.data_dir, f"{dataset_name}/{dataset_name}_test.csv"),
            index=False,
        )


class HateSpeech(TartDataset):
    """Downloads the Hate Speech 18 dataset for TART eval"""

    _domain = "text"
    _hf_dataset_identifier = "hate_speech18"
    _input_key = "text"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        df = dataset["train"].to_pandas()

        # hate_speech18 has 4 classes, we make it binary
        df = df[df["label"].isin([self.neg_class, self.pos_class])]

        # hate_speech18 has no test set
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.75, random_state=self.seed
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key].tolist()
        y_train = self.train_df["label"].tolist()
        X_test = self.test_df[self.input_key].tolist()
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test


class SMSSpam(TartDataset):
    """Downloads the SMS Spam dataset for TART eval"""

    _domain = "text"
    _hf_dataset_identifier = "sms_spam"
    _input_key = "sms"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        df = dataset["train"].to_pandas()

        # hate_speech18 has 4 classes, we make it binary
        df = df[df["label"].isin([self.neg_class, self.pos_class])]

        # hate_speech18 has no test set
        self.train_df, self.test_df = train_test_split(
            df, test_size=0.75, random_state=self.seed
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key].tolist()
        y_train = self.train_df["label"].tolist()
        X_test = self.test_df[self.input_key].tolist()
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test


class SpeechCommands(TartDataset):
    """Downloads the Speech Commands dataset for TART eval"""

    _domain = "audio"
    _hf_dataset_identifier = "speech_commands"
    _input_key = "audio"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def _prep_train_split(self, total_train_samples, seed, k_range):
        dataset_train_pos = self.train_df.filter(
            lambda example: example["label"] == 1
        ).shuffle(seed=self.seed)

        dataset_train_neg = self.train_df.filter(
            lambda example: example["label"] == 0
        ).shuffle(seed=self.seed)

        min_label_cnt = min(len(dataset_train_pos), len(dataset_train_neg))

        assert (
            k_range[-1] / 2 <= min_label_cnt
        ), f"k_range[-1] is too large, max value should be {min_label_cnt * 2}. Please change"

        samples = []
        total_sampled = 0
        for k in self.k_range:
            curr_k = k - total_sampled
            total_sampled = k
            # sample k samples from each class of train_df_samples, and remove the sampled samples from train_df_samples
            indexes = list(range(total_sampled, total_sampled + int(curr_k / 2)))
            k_samples_pos = dataset_train_pos.select(indexes)
            k_samples_neg = dataset_train_neg.select(indexes)
            # get second level index
            samples += [k_samples_pos, k_samples_neg]

        return concatenate_datasets(samples)

    @property
    def get_dataset(self):
        # random permutation of the dataset
        test_df = self.test_df.shuffle(seed=self.seed)
        if self.max_eval_samples:
            dataset_test_pos = self.test_df.filter(
                lambda example: example["label"] == 1
            ).shuffle(seed=self.seed)

            dataset_test_neg = self.test_df.filter(
                lambda example: example["label"] == 0
            ).shuffle(seed=self.seed)

            min_label_cnt = min(len(dataset_test_pos), len(dataset_test_neg))

            total_eval_samples = min(self.max_eval_samples, min_label_cnt * 2)

            # select total_eval_samples in a class balanced way
            dataset_test_pos = (
                self.test_df.filter(lambda example: example["label"] == 1)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            dataset_test_neg = (
                self.test_df.filter(lambda example: example["label"] == 0)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            test_df = concatenate_datasets([dataset_test_pos, dataset_test_neg])

            self.X_test, self.y_test = (
                test_df[self.input_key],
                test_df["label"],
            )
        return self.X_train, self.y_train, self.X_test, self.y_test

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, "v0.01")

        self.train_df = (
            dataset["train"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.test_df = (
            dataset["test"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key]
        y_train = self.train_df["label"]
        X_test = self.test_df[self.input_key]
        y_test = self.test_df["label"]

        return X_train, y_train, X_test, y_test


class YelpPolarity(TartDataset):
    """Downloads the Yelp Polarity dataset for TART eval"""

    _domain = "text"
    _hf_dataset_identifier = "yelp_polarity"
    _input_key = "text"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        self.train_df = dataset["train"].to_pandas()
        self.test_df = dataset["test"].to_pandas()

        self.train_df["label"] = self.train_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )
        self.test_df["label"] = self.test_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key].tolist()
        y_train = self.train_df["label"].tolist()
        X_test = self.test_df[self.input_key].tolist()
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test


class DBPedia14(TartDataset):
    """Downloads the DBPedia_14 dataset for TART eval"""

    _domain = "text"
    _hf_dataset_identifier = "dbpedia_14"
    _input_key = "content"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        self.train_df = dataset["train"].to_pandas()
        self.test_df = dataset["test"].to_pandas()

        self.train_df["label"] = self.train_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )
        self.test_df["label"] = self.test_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key].tolist()
        y_train = self.train_df["label"].tolist()
        X_test = self.test_df[self.input_key].tolist()
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test


class AGNews(TartDataset):
    """Downloads the AG News dataset for TART eval"""

    _domain = "text"
    _hf_dataset_identifier = "ag_news"
    _input_key = "text"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        self.train_df = dataset["train"].to_pandas()
        self.test_df = dataset["test"].to_pandas()

        self.train_df["label"] = self.train_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )
        self.test_df["label"] = self.test_df["label"].apply(
            lambda x: 1 if x == self.pos_class else 0
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key].tolist()
        y_train = self.train_df["label"].tolist()
        X_test = self.test_df[self.input_key].tolist()
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test


class MNIST(TartDataset):
    """Downloads the MNIST dataset for TART eval"""

    _domain = "image"
    _hf_dataset_identifier = "mnist"
    _input_key = "image"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def _prep_train_split(self, total_train_samples, seed, k_range):
        dataset_train_pos = self.train_df.filter(
            lambda example: example["label"] == 1
        ).shuffle(seed=self.seed)

        dataset_train_neg = self.train_df.filter(
            lambda example: example["label"] == 0
        ).shuffle(seed=self.seed)

        min_label_cnt = min(len(dataset_train_pos), len(dataset_train_neg))

        assert (
            k_range[-1] / 2 <= min_label_cnt
        ), f"k_range[-1] is too large, values should be less than {min_label_cnt * 2}. Please change."

        samples = []
        total_sampled = 0
        for k in self.k_range:
            curr_k = k - total_sampled
            total_sampled = k
            indexes = list(range(total_sampled, total_sampled + int(curr_k / 2)))
            k_samples_pos = dataset_train_pos.select(indexes)
            k_samples_neg = dataset_train_neg.select(indexes)
            # get second level index
            samples += [k_samples_pos, k_samples_neg]

        return concatenate_datasets(samples)

    @property
    def get_dataset(self):
        if self.max_eval_samples:
            dataset_test_pos = self.test_df.filter(
                lambda example: example["label"] == 1
            ).shuffle(seed=self.seed)

            dataset_test_neg = self.test_df.filter(
                lambda example: example["label"] == 0
            ).shuffle(seed=self.seed)

            min_label_cnt = min(len(dataset_test_pos), len(dataset_test_neg))

            total_eval_samples = min(self.max_eval_samples, min_label_cnt * 2)

            # select total_eval_samples in a class balanced way
            dataset_test_pos = (
                self.test_df.filter(lambda example: example["label"] == 1)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            dataset_test_neg = (
                self.test_df.filter(lambda example: example["label"] == 0)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            test_df = concatenate_datasets([dataset_test_pos, dataset_test_neg])

            self.X_test, self.y_test = (
                test_df[self.input_key],
                test_df["label"],
            )
        return self.X_train, self.y_train, self.X_test, self.y_test

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        self.train_df = (
            dataset["train"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.test_df = (
            dataset["test"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key]
        y_train = self.train_df["label"]
        X_test = self.test_df[self.input_key]
        y_test = self.test_df["label"]

        return X_train, y_train, X_test, y_test


class CIFAR10(TartDataset):
    """Downloads the cifar-10 dataset for TART eval"""

    _domain = "image"
    _hf_dataset_identifier = "cifar10"
    _input_key = "img"

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int,
        pos_class: int = 0,
        neg_class: int = 1,
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def _prep_train_split(self, total_train_samples, seed, k_range):
        dataset_train_pos = self.train_df.filter(
            lambda example: example["label"] == 1
        ).shuffle(seed=self.seed)

        dataset_train_neg = self.train_df.filter(
            lambda example: example["label"] == 0
        ).shuffle(seed=self.seed)

        min_label_cnt = min(len(dataset_train_pos), len(dataset_train_neg))

        assert (
            k_range[-1] / 2 <= min_label_cnt
        ), f"k_range[-1] is too large, values should be less than {min_label_cnt * 2}. Please change."

        samples = []
        total_sampled = 0
        for k in self.k_range:
            curr_k = k - total_sampled
            total_sampled = k
            indexes = list(range(total_sampled, total_sampled + int(curr_k / 2)))
            k_samples_pos = dataset_train_pos.select(indexes)
            k_samples_neg = dataset_train_neg.select(indexes)
            # get second level index
            samples += [k_samples_pos, k_samples_neg]

        return concatenate_datasets(samples)

    @property
    def get_dataset(self):
        if self.max_eval_samples:
            dataset_test_pos = self.test_df.filter(
                lambda example: example["label"] == 1
            ).shuffle(seed=self.seed)

            dataset_test_neg = self.test_df.filter(
                lambda example: example["label"] == 0
            ).shuffle(seed=self.seed)

            min_label_cnt = min(len(dataset_test_pos), len(dataset_test_neg))

            total_eval_samples = min(self.max_eval_samples, min_label_cnt * 2)

            # select total_eval_samples in a class balanced way
            dataset_test_pos = (
                self.test_df.filter(lambda example: example["label"] == 1)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            dataset_test_neg = (
                self.test_df.filter(lambda example: example["label"] == 0)
                .shuffle(seed=self.seed)
                .select(list(range(int(total_eval_samples / 2))))
            )

            test_df = concatenate_datasets([dataset_test_pos, dataset_test_neg])

            self.X_test, self.y_test = (
                test_df[self.input_key],
                test_df["label"],
            )
        return self.X_train, self.y_train, self.X_test, self.y_test

    def prepare_train_test_split(self):
        dataset_name = self.hf_dataset_identifier
        dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)

        self.train_df = (
            dataset["train"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.test_df = (
            dataset["test"]
            .filter(
                lambda example: example["label"] in [self.pos_class, self.neg_class]
            )
            .map(
                lambda example: {
                    "label": 0 if example["label"] == self.neg_class else 1
                }
            )
        )

        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = self.train_df[self.input_key]
        y_train = self.train_df["label"]
        X_test = self.test_df[self.input_key]
        y_test = self.test_df["label"]

        return X_train, y_train, X_test, y_test

class RedCaps(TartDataset):
    """Downloads the RedCaps dataset for TART eval. This dataset is multi-modal, with images and accompanying captions,
    so we return an list of tuples of (text, image).
    We evaluate against the 'blacksmith' and 'dogpictures' subreddits."""

    _domain = "multi_text_image"
    _hf_dataset_identifier = "red_caps"
    _input_key = ["image", "caption"]  # ignored in return value

    def __init__(
        self,
        total_train_samples: int,
        k_range: List[int],
        seed: int = 42,
        pos_class: int = 0,  # 51 is id for subreddit r/blacksmith
        neg_class: int = 1,  # 126 is id for subreddit r/dogpictures
        cache_dir: str = None,
        data_dir_path: str = None,
        max_eval_samples: int = None,
    ):
        super().__init__(
            total_train_samples,
            k_range,
            seed,
            pos_class,
            neg_class,
            cache_dir,
            data_dir_path,
            max_eval_samples,
        )

        (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        ) = self.prepare_train_test_split()

    def _prep_train_split(self, total_train_samples, seed, k_range):
        dataset_train_pos = self.train_df.loc[self.train_df["label"] == 1].sample(
            frac=1, random_state=self.seed
        )
        dataset_train_pos.reset_index(drop=True, inplace=True)
        dataset_train_neg = self.train_df.loc[self.train_df["label"] == 0].sample(
            frac=1, random_state=self.seed
        )
        dataset_train_neg.reset_index(drop=True, inplace=True)
        min_label_cnt = min(len(dataset_train_pos), len(dataset_train_neg))

        assert (
            k_range[-1] / 2 <= min_label_cnt
        ), f"k_range[-1] is too large, values should be less than {min_label_cnt * 2}. Please change."

        samples = []
        total_sampled = 0
        for k in self.k_range:
            curr_k = k - total_sampled
            total_sampled = k
            indexes = list(
                range(
                    total_sampled, min(total_sampled + int(curr_k / 2), min_label_cnt)
                )
            )
            k_samples_pos = dataset_train_pos.loc[indexes]
            k_samples_neg = dataset_train_neg.loc[indexes]
            # get second level index
            samples += [k_samples_pos, k_samples_neg]

        return pd.concat(samples)

    def prepare_train_test_split(self):
        if not self.data_dir:
            print("Error: set data_dir_path to allow images to be downloaded.")
        dataset_name = self.hf_dataset_identifier

        # the cli tool for this dataset seems to be broken, so I'll just download the dataset manually
        dataset_a = load_dataset(dataset_name, "blacksmith", cache_dir=self.cache_dir)
        dataset_b = load_dataset(dataset_name, "dogpictures", cache_dir=self.cache_dir)

        num_images_to_sample = 400

        # randomly sample n images from each class
        dataset = concatenate_datasets(
            [
                dataset_a["train"]
                .shuffle(seed=self.seed)
                .select(list(range(num_images_to_sample))),
                dataset_b["train"]
                .shuffle(seed=self.seed)
                .select(list(range(num_images_to_sample))),
            ]
        )
        dataset = dataset.to_pandas()

        # replace the "subreddit" column with a label column
        for i in range(len(dataset)):
            if dataset["subreddit"][i] == 51:
                dataset.at[i, "label"] = 0
            elif dataset["subreddit"][i] == 126:
                dataset.at[i, "label"] = 1
            else:
                dataset = dataset.drop(i)

        def download_image(url: str, filepath: os.path, resize: int) -> Image:
            """Download a single image from a url and save it to a filepath, resizing it if necessary."""
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # check if file already exists
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    img = Image.open(f).convert("L")
            else:
                sleep(1.5)  # be nice to the server
                try:
                    response = requests.get(url)
                    if (
                        response.status_code != 200 or "removed.png" in response.url
                    ):  # taken from the cli tool
                        return None
                    img = Image.open(BytesIO(response.content)).convert("L")
                    if resize:
                        img = img.resize((resize, resize))
                    img.save(filepath, "PNG")
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise e
                    return None
            return img

        def download_images(dataset, resize: int) -> Dataset:
            """Download all the images and add them to the dataset."""

            dataset["image"] = None
            print("Downloading RedCaps images (this may take a while)...)")
            for i in tqdm(range(len(dataset))):
                image = download_image(
                    dataset["image_url"][i],
                    f"{self.data_dir}/{dataset_name}/{hashlib.sha256(dataset['image_url'][i].encode()).hexdigest()}.png",
                    resize=resize,
                )
                if image is not None and type(image) == type(Image.Image()):
                    dataset.at[i, "image"] = image
            return dataset

        dataset = download_images(
            dataset, resize=224
        )  # 224x224 is the max size for the vit model - set to none to not resize
        # the model will automatically resize the images to the correct size, but it's better for storage to do it here

        # drop the rows where the image couldn't be downloaded
        for i in range(len(dataset)):
            if dataset["image"][i] is None:
                dataset = dataset.drop(i)

        # drop unnecessary columns
        wanted_columns = ["image", "caption", "label"]
        dataset = dataset[wanted_columns]

        self.train_df, self.test_df = train_test_split(
            dataset, test_size=0.75, random_state=self.seed
        )
        self.train_df = self._prep_train_split(
            self.total_train_samples, self.seed, k_range=self.k_range
        )

        if self.data_dir:
            self._save()

        X_train = list(
            zip(self.train_df["image"].tolist(), self.train_df["caption"].tolist())
        )
        y_train = self.train_df["label"].tolist()
        X_test = list(
            zip(self.test_df["image"].tolist(), self.test_df["caption"].tolist())
        )
        y_test = self.test_df["label"].tolist()

        return X_train, y_train, X_test, y_test

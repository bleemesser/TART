from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import torch
from reasoning_module.models import TransformerModel
from sklearn.decomposition import PCA


class TartReasoningHead:
    def __init__(
        self,
        n_dims: int,
        n_positions: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        n_y: int,
        path_to_pretrained_head: str,
    ):
        self.n_dims = n_dims
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_y = n_y

        self.tart_head = TransformerModel(
            n_dims=n_dims,
            n_positions=n_positions,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_y=n_y,
        )

        tweights = torch.load(path_to_pretrained_head)
        self.tart_head.load_state_dict(tweights, strict=False)
        self.tart_head = self.tart_head.cuda()


class TartEmbeddingLayer:
    domain: str
    embed_type: str

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
    ):
        self.embed_model_name = embed_model_name
        self.num_pca_components = num_pca_components

    def _load_model_tokenizer(self):
        raise NotImplementedError

    def compute_pca_with_whitening(self, X_tr_embed, X_tst_embed):
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(X_tr_embed)
        X_tr_pca_cor = pca.transform(X_tr_embed)
        X_tst_pca_cor = pca.transform(X_tst_embed)

        X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
        X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
        X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

        cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
        X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

        return X_tr_pca_cor_white, X_tst_pca_cor_white

    def embed(self):
        raise NotImplementedError

    def get_domain(self):
        return self.domain

    def get_embed_strategy(self):
        return self.embed_type

    def get_embed_model_name(self):
        return self.embed_model_name


class TartEmbeddingLayerAC(ABC):
    _domain: str
    _embed_type: str
    _hf_model_family: str

    def __init__(
        self,
        embed_model_name: str,
        num_pca_components: int,
    ):
        self.embed_model_name = embed_model_name
        self.num_pca_components = num_pca_components

    @abstractmethod
    def _load_model_tokenizer(self):
        pass

    def _compute_pca_with_whitening(self, X_tr_embed, X_tst_embed):
        pca = PCA(n_components=self.num_pca_components)
        pca.fit(X_tr_embed)
        X_tr_pca_cor = pca.transform(X_tr_embed)
        X_tst_pca_cor = pca.transform(X_tst_embed)

        X_tr_pca_cor_mean = X_tr_pca_cor.mean(axis=0)
        X_tr_pca_cor_m0 = X_tr_pca_cor - X_tr_pca_cor_mean
        X_tst_pca_cor_m0 = X_tst_pca_cor - X_tr_pca_cor_mean

        cov_X_cor = np.cov(X_tr_pca_cor_m0, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_X_cor)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        X_tr_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tr_pca_cor_m0.T).T
        X_tst_pca_cor_white = (eigenvectors @ D @ eigenvectors.T @ X_tst_pca_cor_m0.T).T

        return X_tr_pca_cor_white, X_tst_pca_cor_white

    @abstractmethod
    def embed(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def embed_type(self) -> str:
        return self._embed_type

    @property
    def _hf_model_family(self) -> str:
        return self._hf_model_family

    @property
    def _model_name(self) -> str:
        return self._model_name

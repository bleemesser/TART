import torch

from .registry import EMBEDDING_REGISTRY_AC

"""
This is the wrapper for any multimodal combination of models and input formats that will be passed to the Tart class
in the "BASE_EMBED_MODEL" parameter rather than the string name of a single model. It calls the respective embed
methods of each models and combines them into a single embedding based on a user-defined calculation.
"""


class TartEmbedModule:
    def __init__(
        self,
        model_names: list[str],
        embed_methods: list[str],
        combination_method: str,
        num_pca_components: int = None,
        cache_dir: str = None,
        path_to_finetuned_embed_model=dict[str, str],
    ):
        """
        Provide a list of model names and embed methods, and a combination method to combine the embeddings. The model names must be as they appear in the registry.
        If there are duplicate modalities (text, image, text), specify the model name and embed method for each in the order that they appear in the input list.
        If a fine-tuned model is required, provide the path to the model in the path_to_finetuned_embed_model dict with the model name as the key.
        num_pca_components and cache_dir should be inherited from the Tart class.
        """
        self.model_names = model_names
        self.embed_methods = (
            embed_methods if embed_methods else ["stream"] * len(model_names)
        )

        assert len(self.model_names) == len(
            self.embed_methods
        ), "model_names should correspond to embed_methods"

        self.stripped_model_names = [
            model_names[i].split("/")[-1] for i in range(len(model_names))
        ]
        assert all(
            model in EMBEDDING_REGISTRY_AC.keys() for model in self.stripped_model_names
        ), "model_names must be in registry"

        assert all(
            method in EMBEDDING_REGISTRY_AC[model]
            for model, method in zip(self.stripped_model_names, self.embed_methods)
        ), "embed_methods must be in registry for each model"

        # Once validated, we can get the actual embed methods
        self.embed_methods = [
            EMBEDDING_REGISTRY_AC[model.split("/")[-1]][method](
                embed_model_name=model,
                num_pca_components=num_pca_components,
                cache_dir=cache_dir,
                # path_to_finetuned_embed_model=path_to_finetuned_embed_model[model] if model in path_to_finetuned_embed_model else None,
            )
            for model, method in zip(self.model_names, self.embed_methods)
        ]
        self.combination_method = combination_method

    def combine_embeddings(
        self, embeddings: list[tuple[list[int]]]
    ) -> tuple[list[float]]:
        """
        Combine the embeddings of each modality into a single embedding per input.
        We take in a tuple of lists of tupled embeddings, where each list corresponds to X_test, X_train, y_test, y_train for each modality.
        We only need to combine the X embeddings, so we return a tuple of lists of floats of the same structure as the input.

        example:
        values = [
            ([ # X_test],
            [ # X_train_subset],
            [ # y_train_subset],
            [ # y_test]),
            ([ # X_test],
            [ # X_train_subset],
            [ # y_train_subset],
            [ # y_test]),
            ...
        ]

        return = [
            [ # X_test_combined],
            [ # X_train_subset_combined],
            [ # y_train_untouched],
            [ # y_test_untouched]),
        ]

        """
        if self.combination_method == "average":  # BY FAR the best method
            weights = [1 / len(embeddings)] * len(
                embeddings
            )  # this will give equal weight to each modality, placeholder for now
            out_x_test = embeddings[0][0]
            out_x_train_subset = embeddings[0][1]

            for i in range(len(embeddings))[1:]:
                out_x_test += weights[i] * embeddings[i][0]
                out_x_train_subset += weights[i] * embeddings[i][1]

            out_x_test /= torch.full(out_x_test.shape, len(embeddings))
            out_x_train_subset /= torch.full(out_x_train_subset.shape, len(embeddings))
        elif self.combination_method == "truncate_concat":
            """
            This method concatenates the first n elements of each modality's set, where n is the desired output length divided by the number of modalities.

            Visual:
            [modality1 (18x16), modality2 (18x16), modality3 (18x16)] (18x16x3) -> [modality1[:n] (6x16), modality2[:n] (6x16), modality3[:n] (6x16)] (18x16)

            It will also add an extra few lines to the output from the first modality if the output length is not evenly divisible by the number of modalities.
            """
            test_len = len(embeddings[0][0])
            train_len = len(embeddings[0][1])
            out_x_test = torch.cat(
                [
                    embeddings[i][0][: test_len // len(embeddings)]
                    for i in range(len(embeddings))
                ],
                dim=0,
            )
            out_x_train_subset = torch.cat(
                [
                    embeddings[i][1][: train_len // len(embeddings)]
                    for i in range(len(embeddings))
                ],
                dim=0,
            )
            # if the output length is not divisible by the number of modalities, we add the remainder to the first modality
            out_x_test = torch.cat(
                [out_x_test, embeddings[0][0][: test_len % len(embeddings)]],
                dim=0,
            )
            out_x_train_subset = torch.cat(
                [
                    out_x_train_subset,
                    embeddings[0][1][: train_len % len(embeddings)],
                ],
                dim=0,
            )
            # divide the output length by the number of modalities to get the length of each modality's output, then concatenate that many elements from each modality
            print(embeddings[0][1].shape)
            print(out_x_train_subset.shape)

        elif self.combination_method == "multiply":
            out_x_test = embeddings[0][0]
            out_x_train_subset = embeddings[0][1]
            for i in range(len(embeddings))[1:]:
                out_x_test *= embeddings[i][0]
                out_x_train_subset *= embeddings[i][1]
        return (
            out_x_test,
            out_x_train_subset,
            embeddings[0][2],
            embeddings[0][3],
        )

    def embed(
        self,
        X_test: list[tuple[any]],
        X_train_subset: list[tuple[any]],
        y_train_subset: list[int],
        y_test: list[int],
        k: int,
        seed: int = 42,
    ) -> list[float]:
        """
        Generates embeddings for each single input per tuple in the input list, then combines them into a single embedding per list item.
        """

        # Separate the input into the different modalities
        num_modalities = len(self.model_names)

        # Generate embeddings for each modality, for both the test and train data
        embeddings_raw = []

        for modality in range(num_modalities):
            print(f"Embedding modality {modality}")
            # Get the inputs for the modality
            X_test_modality = [
                x[modality] if isinstance(x, tuple) else x for x in X_test
            ]
            X_train_modality = [
                x[modality] if isinstance(x, tuple) else x for x in X_train_subset
            ]

            # Get the embed method for the modality
            embed_method = self.embed_methods[modality]

            # Generate the embeddings for the modality
            values = embed_method.embed(
                X_test=X_test_modality,
                X_train_subset=X_train_modality,
                y_train_subset=y_train_subset,
                y_test=y_test,
                k=k,
            )
            embeddings_raw.append(values)
        print("Combining embeddings")
        combined_embeddings = self.combine_embeddings(embeddings_raw)
        # print(embeddings_raw[0])
        # Combine the embeddings by the combination method (weighted average, multiplication, etc.)
        # combined_embeddings = self.combine_embeddings(embeddings_raw)

        return combined_embeddings

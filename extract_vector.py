import dataclasses
import torch 
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from Response_without_Privacy.steering import SteeringModel
from Response_without_Privacy.prepare_dataset import SteeringDataset
from Response_without_Privacy.utils import ContrastivePair
from sklearn.decomposition import PCA
from typing import List
import typing
import numpy as np
from tqdm import tqdm
import warnings
import os
import json
from matplotlib import pyplot as plt



@dataclasses.dataclass
class SteeringVector:
    model_type: str
    direction: dict[int, np.ndarray]
    explained_variances: dict[int, float]

    @classmethod
    def extract(
        cls,
        model: PreTrainedModel|SteeringModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: SteeringDataset,
        **kwargs
    ) -> "SteeringVector":
        print("Start extract steering vector...")
        tokenizer.pad_token_id = 0

        directions, explained_variances = extract_representations(
            model,
            tokenizer,
            dataset.formatted_dataset,
            suffixes = dataset.suffixes,
            **kwargs
        )
        return cls(model_type=model.config.model_type,
                   directions=directions,
                   explained_variances=explained_variances)
    
    def save(self, file_path: str):
        dir = os.path.dirname(file_path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": self.explained_variances
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
        print(f"Successfully saved SteeringVector to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        print(f"Loading SteeringVector from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        directions = {int(k):np.array(v) for k, v in data["directions"].items()}
        explained_variances = {int(k): v for k, v in data["explained_variances"].items()}
        print(f"Loaded directions for layers: {list(directions.keys())}")
        print(f"Shape of first direction vector: {next(iter(directions.values())).shape}")
        
        return cls(model_type=data["model_type"], 
               directions=directions, 
               explained_variances=explained_variances)
    

def extract_representations(
        model: PreTrainedModel|SteeringModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs: List[ContrastivePair],
        suffixes: typing.List[typing.Tuple[str, str]] = None,
        hidden_layer_ids: typing.Iterable[int]|None = None,
        batch_size: int = 4,
        method: str = "pca_pairwise",
        save_analysis: bool = False,
        accumulate_last_x_tokens: typing.Union[int, str] = 1,
        **kwargs
    )-> dict[int, np.ndarray]:
    print("Extracting representations...")
    if hidden_layer_ids is None:
        hidden_layer_ids = range(model.config.num_hidden_layers)
    
    if accumulate_last_x_tokens == "all":
        # Accumulate hidden states of all tokens
        print("... accumulating all hidden states")
    elif accumulate_last_x_tokens == "suffix-only":
        print(f"... accumulating suffix-only hidden states")
    else:
        print(f"... accumulating last {accumulate_last_x_tokens} tokens' hidden states")

    n_layers = len(hidden_layer_ids)
    hidden_layer_ids = [i if i >= 0 else n_layers + i for i in hidden_layer_ids] # 
    input_strs = [s for ex in inputs for s in (ex.positive, ex.negative)]

    layer_hiddens = batched_get_hiddens(model, tokenizer, input_strs, hidden_layer_ids, batch_size, accumulate_last_x_tokens, suffixes)
    if save_analysis:
        save_pca_figures(layer_hiddens, hidden_layer_ids, method, inputs)

    directions: dict[int, np.ndarray] = {}
    explained_variances: dict[int, float] = {}

    for layer in tqdm(hidden_layer_ids, desc="Extracting steering vectors from layers"):
        h = layer_hiddens[layer]
        if method == "pca_diff":
            extracted_vectors = h[::2] - h[1::2]  # Positive - Negative
        elif method == "pca_center":
            center = h.mean(axis=0)
            extracted_vectors = h - center
        elif method == "pca_pairwise":  # 文章里用的这个
            center = (h[::2] + h[1::2])/2
            extracted_vectors = h.copy()
            extracted_vectors[::2] -= center
            extracted_vectors[1::2] -= center
        else:
            raise ValueError(f"Unknown method: {method}")
        
        pca_model = PCA(n_components=1, whiten=False).fit(extracted_vectors)
        directions[layer] = pca_model.components_.astype(np.float32).squeeze()
        explained_variances[layer] = pca_model.explained_variance_[0]

        projected_hiddens = project_onto_direction(h, directions[layer])
        # Calculate the mean of positive examples being smaller than negative examples
        positive_smaller_mean = np.mean(
            [
                projected_hiddens[i] < projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        # Calculate the mean of positive examples being larger than negative examples
        positive_larger_mean = np.mean(
            [
                projected_hiddens[i] > projected_hiddens[i + 1]
                for i in range(0, len(inputs) * 2, 2)
            ]
        )
        # If positive examples are smaller on average, flip the direction vector
        if positive_smaller_mean > positive_larger_mean:
            directions[layer] *= -1

    return directions, explained_variances




def project_onto_direction(H, direction):
    """
    Project a matrix H onto a direction vector.

    Args:
        H: The matrix to project.
        direction: The direction vector to project onto.

    Returns:
        The projected matrix.
    """
    # Calculate the magnitude (Euclidean norm) of the direction vector
    mag = np.linalg.norm(direction)
    
    # Assert that the magnitude is not infinite to ensure validity
    assert not np.isinf(mag)
    
    # Perform the projection by multiplying the matrix H with the direction vector
    # Divide the result by the magnitude of the direction vector to normalize the projection
    return (H @ direction) / mag

    

def save_pca_figures(layer_hiddens, hidden_layer_ids, method, output_dir, inputs):
    """
    Save PCA analysis figures for each hidden layer and create a macroscopic x-axis layer analysis plot.

    Args:
        layer_hiddens: A dictionary of hidden states for each layer.
        hidden_layer_ids: The IDs of hidden layers.
        method: The method used for preparing training data.
        output_dir: The directory to save the figures to.
        inputs: The input data used for the analysis.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lists to store the variances and layer IDs for the macroscopic analysis
    variances = []
    layers = []

    for layer in tqdm(hidden_layer_ids, desc="Saving PCA Figures"):
        h = layer_hiddens[layer]

        if method == "pca_diff":
            train = h[::2] - h[1::2]
        elif method == "pca_center":
            center = h.mean(axis=0)
            train = h - center
        elif method == "pca_pairwise":
            center = (h[::2] + h[1::2]) / 2
            train = h.copy()
            train[::2] -= center
            train[1::2] -= center
        else:
            raise ValueError("unknown method " + method)

        pca_model = PCA(n_components=2, whiten=False).fit(train)

        # Project the dataset points onto the first two principal components
        projected_data = pca_model.transform(h)

        # Separate the projected data into positive and negative examples
        positive_data = projected_data[::2]
        negative_data = projected_data[1::2]

        # Plot the projected points with separate colors for positive and negative examples
        plt.figure(figsize=(8, 6))
        plt.scatter(positive_data[:, 0], positive_data[:, 1], alpha=0.7, label="Positive Examples")
        plt.scatter(negative_data[:, 0], negative_data[:, 1], alpha=0.7, label="Negative Examples")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA Visualization - Layer {layer}")
        plt.legend()
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, f"pca_layer_{layer}.png"))
        plt.close()

        # Store the variance explained by PC1 and the corresponding layer ID for the macroscopic analysis
        variances.append(pca_model.explained_variance_ratio_[0])
        layers.append(layer)

    # Create the macroscopic x-axis layer analysis plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, variances, marker='o')
    plt.xlabel("Layer ID")
    plt.ylabel("Variance Explained by PC1")
    plt.title("Macroscopic X-Axis Layer Analysis")
    plt.grid(True)
    plt.xticks(layers)
    plt.tight_layout()

    # Save the macroscopic analysis figure
    plt.savefig(os.path.join(output_dir, "macroscopic_analysis.png"))
    plt.close()

def batched_get_hiddens(
        model,
        tokenizer,
        input_strs,
        hidden_layer_ids,
        batch_size,
        accumulate_last_x_tokens,
        suffixes
    )-> dict[int, np.ndarray]:
    """
    Retrieve the hidden states from the specified layers of the language model for the given input strings.

    Args:
        model: The model to get hidden states from.
        tokenizer: The tokenizer associated with the model.
        inputs: A list of input strings.
        hidden_layer_ids: The IDs of hidden layers to get states from.
        batch_size: The batch size to use when processing inputs.
        accumulate_last_x_tokens: How many tokens to accumulate for the hidden state.
        suffixes: List of suffixes to use when accumulating hidden states.

    Returns:
        A dictionary mapping layer IDs to numpy arrays of hidden states.
    """
    batched_inputs  = [
        input_strs[p: p + batch_size] for p in range(0, len(input_strs), batch_size)
    ]
    hidden_states = {layer: [] for layer in hidden_layer_ids}

    with torch.no_grad():
        for batch in tqdm(batched_inputs, desc="Getting hidden states:"):
            tokenized = tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
            )
            outputs = model(
                **{k:v.to(model.device) for k,v in tokenized.items()},
                output_hidden_states = True  # Ensure hidden states are returned
            )

            for layer_id in hidden_layer_ids:
                hidden_idx = layer_id + 1 if layer_id >=0 else layer_id # 这里为社么加1
                for i, batch_hidden in enumerate(outputs.hidden_states[hidden_idx]):
                    if accumulate_last_x_tokens == "all":
                        accumulated_hidden_state = torch.mean(batch_hidden, dim=0)
                    elif accumulate_last_x_tokens == "suffix-only":
                        if suffixes:
                            suffix_tokens = tokenizer.encode(suffixes[0][0], add_special_tokens=False)
                            suffix_hidden = batch_hidden[-len(suffix_tokens):, :]
                            accumulated_hidden_state = torch.mean(suffix_hidden, dim=0)
                        else:
                            warnings.warn("'suffix-only' option used but no suffixes provided. Using last token instead.")
                            accumulated_hidden_state = batch_hidden[-1, :]
                    else:
                        accumulated_hidden_state = torch.mean(batch_hidden[-accumulate_last_x_tokens:, :], dim=0)

                    hidden_states[layer_id].append(accumulated_hidden_state.cpu().numpy())
            del outputs

        return {k:np.vstack(v) for k,v in hidden_states.items()}
                    
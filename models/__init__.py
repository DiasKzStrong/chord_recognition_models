from .btc import BTCChordModel, StructuredBTCChordModel
from .htv2 import HTv2ChordModel, HyperParameters, StructuredHTv2ChordModel


def build_model(
    model_type: str,
    input_dim: int,
    vocab,
    hyperparameters: HyperParameters,
    dropout_rate: float,
):
    if getattr(vocab, "label_mode", "") == "structured_full_chord":
        component_sizes = {
            name: len(vocab.component_labels[name])
            for name in vocab.component_names
        }
        if model_type == "htv2":
            return StructuredHTv2ChordModel(
                input_dim=input_dim,
                component_sizes=component_sizes,
                chord_component_ids=vocab.chord_component_ids,
                hyperparameters=hyperparameters,
                dropout_rate=dropout_rate,
            )
        if model_type == "btc":
            return StructuredBTCChordModel(
                input_dim=input_dim,
                component_sizes=component_sizes,
                chord_component_ids=vocab.chord_component_ids,
                hyperparameters=hyperparameters,
                dropout_rate=dropout_rate,
            )
    else:
        if model_type == "htv2":
            return HTv2ChordModel(
                input_dim=input_dim,
                n_chords=vocab.size,
                hyperparameters=hyperparameters,
                dropout_rate=dropout_rate,
            )
        if model_type == "btc":
            return BTCChordModel(
                input_dim=input_dim,
                n_chords=vocab.size,
                hyperparameters=hyperparameters,
                dropout_rate=dropout_rate,
            )

    raise ValueError(f"Unsupported model_type: {model_type}")

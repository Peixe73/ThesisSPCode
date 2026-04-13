import torch
import numpy as np
from PIL import Image

from core import datasets
from scripts.studies.gtsrb.hn_1 import ENTRY_CONCEPTS
from models.conv_network import make_model


THRESHOLD = 0.5


def load_model(model_path, dataset):
    input_shape = dataset.get_shape()[0]

    model = make_model(
        input_shape=input_shape,
        conv_layers=[32, 32, ('pool', 2), 64, ('pool', 2), 128, ('pool', 2)],
        linear_layers=[128],
        num_outputs=len(ENTRY_CONCEPTS),
        hidden_activations=('leaky_relu', 0.1)
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def inspect_sample(idx, model, dataset):

    x, y_true = dataset[idx]

    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs = torch.sigmoid(logits)[0]

    preds = (probs > THRESHOLD).int()

    true = y_true.int()

    print("\n=== DEBUG SAMPLE ===")

    for i, concept in enumerate(ENTRY_CONCEPTS):
        if preds[i] == 1 and true[i] == 1:
            status = "✔ TP"
        elif preds[i] == 1 and true[i] == 0:
            status = "✖ FP"
        elif preds[i] == 0 and true[i] == 1:
            status = "✖ FN"
        else:
            continue

        print(f"{concept:30} | pred={preds[i].item()} true={true[i].item()} | {status}")


def main():
    dataset = datasets.get_dataset("gtsrb_concepts_only")

    model = load_model("best_model.pt", dataset)

    # Pick any index
    inspect_sample(idx=10, model=model, dataset=dataset)


if __name__ == "__main__":
    main()
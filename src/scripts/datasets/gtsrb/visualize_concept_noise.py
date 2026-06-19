'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_tools.random_reasoning_dataset import RandomReasoningDataset
from analysis_tools.gtsrb_utils import CONCEPTS, CLASS_COLS

MODES = ["binary", "uniform", "extremes", "middle"]
N_SAMPLES = 50000

COLORS = {
    "binary": "tab:blue",
    "uniform": "tab:orange",
    "extremes": "tab:green",
    "middle": "tab:red",
}

def generate_mode_samples(mode, seed=42):
    ds = RandomReasoningDataset(
        valid_path="data/gtsrb_ontology_Valid_Final.csv",
        feature_cols=CONCEPTS,
        class_cols=CLASS_COLS,
        dataset_size=10,
        seed=seed,
        concept_noise_mode=mode
    )

    rng = np.random.default_rng(seed)

    base = rng.integers(
        0, 2,
        size=(N_SAMPLES, len(CONCEPTS))
    ).astype(np.float32)

    noisy = np.zeros_like(base)

    for i in range(N_SAMPLES):
        noisy[i] = ds._soften_features(base[i])

    return noisy.flatten()


def plot_modes_kde():
    plt.figure(figsize=(10, 6))

    for mode in MODES:
        print(f"Generating {mode}...")
        data = generate_mode_samples(mode)

        sns.kdeplot(
            data,
            label=mode,
            color=COLORS[mode],
            linewidth=2
        )

    plt.title("Concept Noise Modes (KDE)")
    plt.xlabel("Concept value")
    plt.ylabel("Density")
    plt.xlim(0, 1)

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "CodeForThesisImages/Images/concept_noise_modes_kde.png",
        dpi=300
    )
    plt.show()


def main():
    plot_modes_kde()

if __name__ == "__main__":
    main()

'''
import numpy as np
import matplotlib.pyplot as plt
from analysis_tools.random_reasoning_dataset import RandomReasoningDataset
from analysis_tools.gtsrb_utils import CONCEPTS, CLASS_COLS

MODES = ["binary", "uniform", "extremes", "middle"]

N_SAMPLES = 1000000
#N_FEATURES = 53  # or len(CONCEPTS)

def generate_mode_samples(mode, seed=42):
    ds = RandomReasoningDataset(
        valid_path="data/gtsrb_ontology_Valid_Final.csv",
        feature_cols=CONCEPTS, #[f"f{i}" for i in range(N_FEATURES)],
        class_cols=CLASS_COLS,#["dummy"],
        dataset_size=10,
        seed=seed,
        concept_noise_mode=mode
    )

    rng = np.random.default_rng(seed)

    # start from clean binary features
    base = rng.integers(0, 2, size=(N_SAMPLES, len(CONCEPTS))).astype(np.float32)

    noisy = np.zeros_like(base)

    for i in range(N_SAMPLES):
        noisy[i] = ds._soften_features(base[i])

    #return noisy.flatten()
    return noisy.reshape(-1)

def plot_modes():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, mode in zip(axes, MODES):
        print(f"Generating {mode}...")
        data = generate_mode_samples(mode)

        ax.hist(data, bins=100, density=True)
        ax.set_title(f"{mode.capitalize()} Noise")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(
        "CodeForThesisImages/Images/concept_noise_modes.png",
        dpi=300
    )
    plt.show()
    
def plot_modes_separately():
    for mode in MODES:
        print(f"Generating {mode}...")
        data = generate_mode_samples(mode)

        plt.figure(figsize=(5, 4))
        plt.hist(data, bins=100, density=True)

        plt.title(f"Concept noise: {mode}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.xlim(0, 1)

        plt.tight_layout()
        plt.savefig(
            f"CodeForThesisImages/Images/concept_noise_{mode}.png",
            dpi=300
        )
        plt.close()

"""
def plot_modes():
        plt.figure()

        for mode in MODES:
            print(f"Generating {mode}...")
            data = generate_mode_samples(mode)
            plt.hist(data, bins=100, alpha=0.5, density=True, label=mode)

        plt.title("Concept Noise Mode Distributions")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.savefig("CodeForThesisImages/Images/concept_noise_modes.png", dpi=300)
        plt.show()
"""

def main():
    plot_modes()

if __name__ == "__main__":
    main()
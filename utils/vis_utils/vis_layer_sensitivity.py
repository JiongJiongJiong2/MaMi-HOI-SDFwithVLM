"""Minimal plotting helper used by the training/evaluation entrypoint."""


def plot_geometric_dilution(results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(results.get("layers", []))
    text_corr = list(results.get("text_corr", []))
    bps_corr = list(results.get("bps_corr", []))
    if not layers:
        return

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(layers, text_corr, marker="o", label="text correlation")
    axis.plot(layers, bps_corr, marker="o", label="BPS correlation")
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Mean cosine similarity")
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=150)
    plt.close(figure)

# -*- coding: utf-8 -*-
"""Script for making boxplot.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig


def main(cfg: DictConfig):
    """Plot boxplot of scores."""
    score_dir = os.path.join(cfg.vM.root_dir, cfg.vM.score_dir)
    score_file = {"Amp": None, "Rnd": None, "vM": None}
    score = {"Amp": None, "Rnd": None, "vM": None}
    fig_dir = os.path.join(cfg.vM.root_dir, cfg.vM.fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    for i, mode in enumerate(("stoi", "pesq", "lsc")):
        gla_iter = cfg.feature.gla_iter
        if cfg.demo.gla is False:
            score_file["Amp"] = os.path.join(score_dir, f"{mode}_score_zero.txt")
            score_file["Rnd"] = os.path.join(score_dir, f"{mode}_score_random.txt")
            score_file["vM"] = os.path.join(score_dir, f"{mode}_score_0.txt")
        else:
            score_file["Amp"] = os.path.join(
                score_dir, f"{mode}_score_{gla_iter}_zero.txt"
            )
            score_file["Rnd"] = os.path.join(
                score_dir, f"{mode}_score_{gla_iter}_random.txt"
            )
            score_file["vM"] = os.path.join(score_dir, f"{mode}_score_{gla_iter}.txt")
        with open(score_file["Amp"], mode="r", encoding="utf-8") as file_hander:
            score["Amp"] = np.array([float(line.strip()) for line in file_hander])
        with open(score_file["Rnd"], mode="r", encoding="utf-8") as file_hander:
            score["Rnd"] = np.array([float(line.strip()) for line in file_hander])
        with open(score_file["vM"], mode="r", encoding="utf-8") as file_hander:
            score["vM"] = np.array([float(line.strip()) for line in file_hander])

        axes = fig.add_subplot(1, 3, i + 1)
        axes.boxplot(
            np.concatenate(
                (
                    score["Amp"].reshape(-1, 1),
                    score["Rnd"].reshape(-1, 1),
                    score["vM"].reshape(-1, 1),
                ),
                axis=1,
            ),
            flierprops=dict(marker="+", markeredgecolor="r"),
            labels=["Amp", "Rnd", "vM"],
            widths=(0.5, 0.5, 0.5),
        )
        axes.xaxis.set_ticks_position("both")
        axes.yaxis.set_ticks_position("both")
        if mode == "pesq" and cfg.demo.gla is True and gla_iter == 100:
            axes.set_yticks([3, 3.5, 4.0, 4.5])
        elif mode == "pesq" and cfg.demo.gla is True and gla_iter == 10:
            axes.set_yticks([2.5, 3.0, 3.5])
            axes.set_ylim([2.2, 3.7])
        axes.tick_params(direction="in", labelsize=16)
        if mode == "lsc":
            axes.set_title(mode.upper() + " [dB]", fontsize=16)
        else:
            axes.set_title(mode.upper(), fontsize=16)
    fig.tight_layout()
    if cfg.demo.gla is False:
        plt.savefig(os.path.join(fig_dir, "score.png"))
    else:
        plt.savefig(os.path.join(fig_dir, f"score_gla{gla_iter}.png"))
    plt.show()


if __name__ == "__main__":
    with initialize(version_base=None, config_path="."):
        config = compose(config_name="config")
    main(config)

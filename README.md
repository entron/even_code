# Efficient Representation of Natural Image Patches

<p align="center">
  <img src="2d_gaussian.png" alt="2D Gaussian visualization" height="220" />
  <img src="feature_maps_pca.png" alt="Feature maps PCA visualization" height="220" />
  <img src="color.png" alt="Color image visualization" height="220" />
  <img src="spiketrains3.png" alt="Spike trains visualization" height="220" />
</p>

This repository contains code to reproduce the main experiments from:

> *Efficient Representation of Natural Image Patches*  
> [https://arxiv.org/abs/2210.13004](https://arxiv.org/abs/2210.13004)

Since 2011, I've been exploring the following question in my spare time—mostly as an intellectual challenge, and just for fun:

> *If you were an engineer tasked with designing the visual system of an organism, what would you do?*

Why? Fundamentally because I want to understand how our brains process information—and because:

> **“What I cannot create, I do not understand.”**  
> — Richard Feynman

There are too many intricacies in biological systems.
One can easily get lost in the dendritic forests or stuck in the ion channels.
So I kept asking: What are the fundamental goals of information processing? Are there principles that help us extract the essentials from the complexities?

The question itself is too big. It’s tempting to simplify by making many assumptions.
But what Poisson, Gauss, and Bayes can’t tell you is: while making a problem calculable, are you also making your model unrealistic?
So I approached it by imagining a minimal system: first with just one input pixel, then two, and eventually small image patches—always trying to make as few assumptions as possible. I wanted to see how far I could get.

A checkpoint in this decade-long, exciting, and sometimes lonely journey is summarized in the note above.


## Repository structure

- `two_pixel/` – Jupyter notebooks for the two‑pixel case analyzed in the paper. See `two_pixel/README.md`.
- `image_patch/` – code and scripts for training and analyzing the image-patch case in the paper. See `image_patch/README.md` for details.
- `requirement.txt` – Python dependencies for running the experiments and notebooks.


## Two directions I am looking for collaborators

There are two directions I would be very happy to find collaborators to work on.

### EvenCodeTokenizer

Goal: turn the IPU-based binary representation into a practical **image / video tokenizer** for downstream models (e.g., vision transformers, multimodal LLMs, robotics pipelines (VLAs)).

If you are interested in making the tokenizer more robust and benchmarking it, and you have access to GPUs, please email me.


### Comparison with real neural recordings

Goal: systematically compare the IPU’s binary population codes with **real neural data** from early visual areas (retina, LGN, V1, V2, etc.).

If you work with neural data (or have access to relevant datasets) and are interested in testing the theory against real recordings, please contact me.


## Citation

If you use this code or build on these ideas, please cite:

```bibtex
@misc{guo2024efficientrepresentationnaturalimage,
      title         = {Efficient Representation of Natural Image Patches},
      author        = {Cheng Guo},
      year          = {2024},
      eprint        = {2210.13004},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2210.13004},
}
```

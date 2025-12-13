## Two-pixel cases

This folder contains Jupyter notebooks used to study the two-pixel case in the the paper  
[Efficient Representation of Natural Image Patches](https://arxiv.org/abs/2210.13004) (see Fig. 2 and the loss functions in Eqs. (16) and (20)).

### Notebooks

- `generate_gray_image_patches.ipynb`  
  Samples pairs of pixels from natural images (converted to grayscale on the fly).  
  These empirical two-pixel pairs are used to build the distributions behind Fig. 2(a) and Fig. 2(c).

- `generate_artificial_two_pixels.ipynb`  
  Constructs an artificial two-dimensional distribution used for the synthetic example in Fig. 2(b).

- `maximized_entropy_representation_of_image_patches.ipynb`  
  Learns a two-dimensional representation of image patches by optimizing the maximum-entropy losses defined in Eqs. (16) and (20).  
  Using the two-pixel pairs saved by one of the `generate_*.ipynb` notebooks above, this notebook can be used to reproduce the plots in Fig. 2.

### Typical workflow

1. Run one (or both) of the data-generation notebooks:
   - `generate_gray_image_patches.ipynb` for real-image two-pixel pairs.
   - `generate_artificial_two_pixels.ipynb` for the synthetic distribution.
2. Run `maximized_entropy_representation_of_image_patches.ipynb` to train the model and visualize the learned 2D representation and the corresponding partitions/contours shown in Fig. 2.


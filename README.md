# ViTAL SCENE: Vision Transformer-Based Triplet Autoencoder Latent Space for Traffic Scenario Novelty Estimation

This repository provides the triplet autoencoder architecture using vision transformer as encoder as presented in "Traffic Scenario Novelty Detection in the Latent Space of a Vision Transformer-Based Autoencoder Triplet Network".

## Setup and Train the Network
1. Clone the repository.
2. Edit `cu101` in `requirements.txt` to match your cuda version (remove if 10.2).
3. Run `pip install requirements.txt` to install the dependencies. This implementation is based on on pyTorch.
4. Download the data from https://faubox.rrze.uni-erlangen.de/getlink/fiWu4mseP8uEnNNzJJ2dTqtR/Data.zip and unzip it into a folder `\Data`, rooted in the project directory. 
5. Run `python main.py` to train the model. Adjust parameters as required in `main.py` and `scenenet.py`.

## Sources
The files `dataset.py`, `scenenet.py`, `training.py` and `main.py` are based on the implementation in [tile2vec] (https://github.com/ermongroup/tile2vec).The implementation is changed to suit the application, given the infrastructure images and the according graph-IDs. Furthermore, the autoencoder scheme is adopted. The Vision-Transformer implementation is realized through `vit_pytorch.py`, provided in [vit-pytorch] (https://github.com/lucidrains/vit-pytorch).


## Citation
If you are using this repository, please cite the work
```
@InProceedings{Wurst2021a,
  author    = {Jonas {Wurst} and Lakshman {Balasubramanian} and Michael {Botsch} and Wolfgang {Utschick}},
  title     = {Traffic Scenario Novelty Detection in the Latent Space of a Vision Transformer-Based Autoencoder Triplet Network},
  booktitle = {2021 IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2021},
}
```

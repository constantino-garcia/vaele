# SDE-SVAE (Aka Vaele)
This code implements a Structured Variational AutoEncoder (SVAE) that permits 
to find a latent space described by Stochastic Differential Equations (SDE, also known as 
Langevin equations). Note that this enables finding Markovian representations of time series with 
memory and also tackling the important problem of [phase-space reconstruction](http://www.scholarpedia.org/article/Attractor_reconstruction).

Further details may be found in this [Arxiv preprint](http://arxiv.org/abs/2010.06265).

## Getting started
To analyze a new time series you must save your data as a CSV file, which should be further split in train and test. 
Each row should represent a temporal instant, whereas each column should represent each of the variables of your time series. 
The CSV can be then transformed to TFRecord format for fast training (see the `experiment/settings/oscillator.py` example).

Then, you must create an `Experiment` in the `experiment/settings` folder. This is used
for tuning the some of the parameters of the experiment. See  `oscillator` for a self-contained example. The most important 
parameters are the embedding dimension (`embedding_dim`, the dimension of the latent space to be discovered) and the paths to 
the TFRecord folders.

The latent space and the SDEs describing its dynamical features are learnt
running the `run_experiment.py` script. The experiment to be run is loaded
in the import section of the code, i.e:

```python
# Load the experiment
from experiments.settings.oscillator import experiment, svae_settings
# ...
```
The variational autoencoder and the results of the inference procedure are
stored in the `tensorboard_dir` folder configured as part of the experiment.

### An example: the Lotka-Volterra stochastic model
In this example, we illustrate the use of the SDE-based VAE to a 
nonlinear stochastic model inspired by the Lotka-Volterra equations. 
These equations are used in the study of biological ecosystems
to describe the interactions between a predator and its prey. 

The next Gif shows the evolution of the latent phase space after 1, 10,
25, 55 and 450 iterations. The mean and covariances of the encoding network
are represented by orange lines and gray ellipses, respectively. A single 
illustrative sample of the phase space is shown with a dashed blue line. 
Finally, the inducing-inputs are represented with reddish points, whereas 
the drift is represented by arrows. Note that the latent phase space quickly
captures the oscillatory nature of the Lotka-Volterra model. 

<p align="center">
<img src="figs/lotka_volterra/phase_space.gif" alt="Lotka-Volterra phase space" width=400>
</p>

From this latent phase space, is possible to reproduce the exact time series
that feeds the training of the VAE. During the training process, the decoder
tries to reproduce as exactly as possible the input time series, driving the
optimization process of the net. This is illustrated by the next Gif:

<p align="center">
<img src="figs/lotka_volterra/output.gif" alt="Lotka-Volterra output" width=400>
</p>

Once the VAE has finished its training, it may be used for generating new
synthetic samples, as illustrated below. The top panel
shows real data generated from a Lotka-Volterra model. Since the model is 
stochastic, two different realizations are shown. The bottom panel shows 
two synthetic time series (blue and orange) generated by the SVAE given the
data up to the vertical line, colored in black.

<p align="center">
<img src="figs/lotka_volterra/lokta_volterra_reconstruction-1.png" alt="Lotka-Volterra reconstruction" width=400>
</p>

## Dependencies
* TensorFlow > 2.0
* GPFlow > 2.1.1




## Deep learning for Hidden Markov model 

In this project, we study how we can use deep learning techniques in Hidden Markov model for analyzing time-series data. 

References:

- Paper: [Krishnan, Rahul, Uri Shalit, and David Sontag. "Structured inference networks for nonlinear state space models." In Proceedings of the AAAI conference on artificial intelligence, vol. 31, no. 1. 2017.](https://ojs.aaai.org/index.php/AAAI/article/view/10779)
- Code: [deepHMM](https://github.com/guxd/deepHMM)

### Dataset

We modify the simulated dataset from the original study. 
Application example of simulated data - A drug treatment study where we have data in `cell-line x time x cnv` format, for example `20 x 160 x 88` where 88 is pseudo CNV profile from cells in 160 time-points treated with a drug in 20 different cell lines. Our interest is to understand how CNV profile changes i.e. given a profile at time $t$ we want to predict changes at time $t+1$.


### Original work is:

[deepHMM](https://github.com/guxd/deepHMM)

### DHMM
A PyTorch implementation of a Deep Hidden Markov Model: [Structured Inference Networks for Nonlinear State Space Models](https://arxiv.org/pdf/1609.09869.pdf).

Adopted from https://github.com/uber/pyro/tree/dev/examples/dmm  
         and https://github.com/clinicalml/structuredinference


#### Related Work

- [Structured Inference Networks for Nonlinear State Space Models](https://arxiv.org/pdf/1609.09869.pdf)
- [2019ICLR - A NOVEL VARIATIONAL FAMILY FOR HIDDEN NON-LINEAR MARKOV MODELS](https://openreview.net/pdf?id=SJMO2iCct7)
- [A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/pdf/1506.02216.pdf)
- [STATE SPACE LSTM MODELS WITH PARTICLE MCMC INFERENCE](https://arxiv.org/pdf/1711.11179.pdf)
- [BLACK BOX VARIATIONAL INFERENCE FOR STATE SPACE MODEL](Shttps://arxiv.org/pdf/1511.07367.pdf)
- [Gaussian variational approximation for high-dimensional state space models](https://arxiv.org/pdf/1801.07873.pdf)
- [Generating Long-term Trajectories Using Deep Hierarchical Networks](http://papers.nips.cc/paper/6520-generating-long-term-trajectories-using-deep-hierarchical-networks.pdf)
- [Disentangled Sequential Autoencoder](https://arxiv.org/pdf/1803.02991.pdf)

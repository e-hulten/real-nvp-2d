# Real NVP 
Basic implementation of the Real NVP paper by Dinh et al. (https://arxiv.org/abs/1605.08803). This implementation currently only supports 2D densities.  

Change the relevant parameters in `train.py` and run. E.g.,
```
# ------------ parameters ------------
continue_training = False 
moons = True 
n_c_layers = 8 
epochs = 200 
batch_size = 256 
lr = 1e-4 # set the learning rate of the adam optimiser
path = r'/Users/edvardhulten/real_nvp_2d/' # change to your own path 
# ------------------------------------
```
should yield good results on the two moons dataset in a very reasonable amount of time.

Run `eval.py` to plot the results. 

import torch 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import RealNVP
from loss import NegLogLik
from data_loader import Dataset
import time

# ------------ parameters ------------
continue_training = True # set to false if you want to train a new model 
moons = False # set to true if you want to use the two moons dataset
n_c_layers = 12 # number of coupling layers
epochs = 50 # number of epochs to train for
batch_size = 256 # set batch size
lr = 1e-4 # set the learning rate of the adam optimiser
path = r'/Users/edvardhulten/real_nvp_2d/' # change to your own path (unless your name is Edvard Hultén too)
# ------------------------------------

data = Dataset(moons)
x = data.generate_data()

model = RealNVP(n_c_layers)
if continue_training is True:
    model.load_state_dict(torch.load((path + 'model.pt')))

loss_log_det_J = NegLogLik(model)
losses = []

train_loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
start = time.time()
for epoch in range(epochs):
    for x_batch in train_loader:
        model.train()    
        z, sum_log_det_J = model(torch.from_numpy(x_batch.numpy()))
        loss = loss_log_det_J(z, sum_log_det_J)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        stop = time.time()
        print('(epoch %s/%s) loss : %.3f' % (epoch, epochs, loss.item()))
        print('Time lapsed for previous 10 epochs:',stop-start)
        start = stop
        # test
        if epoch % 10 == 0:
            if not os.path.exists('results'):
                os.makedirs('results')
            
            model.eval()
            z, _ = model(torch.from_numpy(x_batch.numpy()))
            z = z.detach().numpy()
            plt.scatter(z[:,0], z[:,1])
            plt.savefig('results/'+'result_'+str(epoch)+'.png')
            plt.clf()

torch.save(model.state_dict(), (path+'model.pt'))
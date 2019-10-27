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
n_c_layers = 32 # number of coupling layers
epochs = 200 # number of epochs to train for
batch_size = 512 # set batch size
lr = 5e-4 # set the learning rate of the adam optimiser
lr_decay = 0.999
path = r'/Users/edvardhulten/real_nvp_2d/' # change to your own path (unless your name is Edvard Hultén too)
# ------------------------------------

if not os.path.exists('results'):
    os.makedirs('results')

data = Dataset(moons)
x = data.generate_data()

model = RealNVP(n_c_layers)

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

if continue_training is True:
    checkpoint = torch.load(path + 'model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler'])

    tot_epochs = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    tot_epochs = 0


loss_log_det_J = NegLogLik(model)

train_loader = DataLoader(dataset=x, batch_size=batch_size, shuffle=True)

start = time.time()
for epoch in range(tot_epochs+1,tot_epochs+epochs+1):
    for x_batch in train_loader:
        model.train()    

        z, sum_log_det_J = model(torch.from_numpy(x_batch.numpy()))
        loss = loss_log_det_J(z, sum_log_det_J)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    if epoch % 10 == 0:
        stop = time.time()
        print('(epoch %s/%s) loss : %.3f' % (epoch, tot_epochs+epochs, loss.item()))
        print('time elapsed previous 10 epochs: {0:.2f} sec'.format(stop-start))
        start = stop
        # test
        if epoch % 10 == 0:    
            model.eval()
            z, _ = model(torch.from_numpy(x_batch.numpy()))
            z = z.detach().numpy()
            plt.scatter(z[:,0], z[:,1])
            plt.savefig('results/'+'result_'+str(epoch)+'.png')
            plt.clf()


torch.save({'epoch': tot_epochs + epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            #'scheduler': scheduler.state_dict()
            }, (path + 'model.pt'))
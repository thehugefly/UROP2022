import torch
import os
import spintorch
import numpy as np
import idx2numpy
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot
import matplotlib.pyplot as plt
import warnings
import sys
#import random

warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")    



'''Directories and parameters to change each time'''

basedir = 'custom_spin_ice/' #####

plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    
    
dev = torch.device('cuda')  # 'cuda' or 'cpu' #####
print('Running on', dev)

epoch_number = 1 #####
epoch_init = -1 #####
loss_T = 0          # threshold of loss, not used yet
#random.seed(10)

print('epoch_init:', epoch_init)
print('epoch_number:', epoch_number)
print('loss_T', loss_T)

n_writing = 2 ##### number of writings of each digit from 0 to 9
print('number of writings', n_writing)

Np = 10  # number of probes #####
probe_size = 2 #####

OUTPUTS_list = [0,1,2,3,4,5,6,7,8,9]

#OUTPUTS_list = [1,3,5,7,9,11,13,15,17,19] #####


'''input and output'''

def get_mnist_image(ID):
    file = 'train-images.idx3-ubyte'
    arr = idx2numpy.convert_from_file(file)
    return(arr[ID])

def get_mnist_label(ID):
    file = 'train-labels.idx1-ubyte'
    label = idx2numpy.convert_from_file(file)
    return(label[ID])

def flatten_image(image,threshold):
    image_new = np.zeros((len(image),len(image[0])))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j]<threshold:
                image_new[i][j] = -1
            else:
                image_new[i][j] = 1
    return(image_new)


ID_list = []
label_list = []

for label in range(0,10):
    ID_list_label = []       # IDs for one label
    
    for ID in range(0, 50):
        label_list.append(get_mnist_label(ID)) # get labels for a range of IDs
    
        if label_list[ID] == label:
            ID_list_label.append(ID)   # get IDs for one label
    
    ID_list_label = ID_list_label[:n_writing] # select same number of IDs for each label
    ID_list.append(ID_list_label)   # list of IDs for all labels specifically    

    
# ID_list = [[1],[3],[5],[7],[2],[0],[13],[15],[17],[4]]

print('ID_list', ID_list)

OUTPUTS = torch.tensor(OUTPUTS_list).to(dev)
print('desired outputs', OUTPUTS_list)

# get image of each ID
image = [[],[],[],[],[],[],[],[],[],[]]

for i in range(0, len(ID_list)):
    for j in range(0, len(ID_list[i])):
        ID = ID_list[i][j]
        digit = get_mnist_image(ID)
        flat_image = flatten_image(digit, 128)
        rot_image = np.rot90(flat_image, k=3, axes=(0, 1)).copy()
        rho_image = torch.tensor(rot_image)
        image[i].append(rho_image)
        

"""Other Parameters"""

frequency = 5.5e9
Bt = 2.5e-3
learning_rate = 0.05

print('frequency:',frequency)
print('Bt:', Bt)
print('learning_rate:', learning_rate)

dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 142+4*60+15      # size x    (cells)
ny = 141       # size y    (cells)

Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
Bt = Bt       # excitation field amplitude (T)
f1 = frequency       # source frequency (Hz)
dt = 20e-12     # timestep (s)

timesteps = 2500 # number of timesteps for wave propagation



'''Original geometry

Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
r0, dr, dm, z_off = 15, 4, 2, 10  # starting pos, period, magnet size, z distance
#rx, ry = int(((nx/2)-2*r0)/dr), int((ny-2*r0)/dr+1)
rx, ry = 28, 28

rx_train,ry_train = 60,28
r0_train= 142 ## Starting x point of the trainable array = 
train_array = np.zeros((60,28)) ## Create trainable array
rho_train = torch.tensor(train_array)
geom = spintorch.geom.WaveGeometryArray_draw_and_train_x_multi(rho_train,(nx, ny), (dx, dy, dz), Ms, B0, 
                                    r0, dr, dm, z_off, rx, ry, r0_train,rx_train,ry_train,Ms_CoPt)

'''

'''Spin ice geometry '''
Ms_Py = 750e3 # saturation magnetization of the nanomagnets (A/m)
r0, dr, wm, lm, z_off = 20, 10, 2, 6, 5  # starting pos, period, magnet size, z distance
rx, ry = int((nx-2*r0)/dr+1), int((ny-2*r0)/dr+1)
rho1 = torch.zeros((rx, ry))  # Design parameter array
rho2 = torch.zeros((rx-1, ry+1))  # Design parameter array
geom = spintorch.geom.WaveGeometrySpinIce(rho1, rho2, (nx, ny), (dx, dy, dz), Ms, B0, 
                                    r0, dr, wm, lm, z_off, rx, ry, Ms_Py)



probes = []

for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-15, int((ny-20)*(p+1)/(Np+1))+10, probe_size))

src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=2)
model = spintorch.MMSolver(geom, dt, [src], probes)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude
INPUTS = X  # here we could cat multiple inputs
print(INPUTS)

'''Define optimizer and lossfunction'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) ## Change this learning rate

def my_loss(output, target_index):
    target_value = output[:,target_index]
    loss = output.sum(dim=1)/target_value-1
    return (loss.sum()/loss.size()[0]).log10()

'''Load checkpoint'''
epoch = epoch_init # select previous checkpoint (-1 = don't use checkpoint)

if epoch_init>=0:
    checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []
    
    
    
    
'''Train the network'''


print('----------------train-------------------')

tic()
model.retain_history = True

for epoch in range(epoch_init+1, epoch_init+epoch_number+1):
    loss_batch = 0

    print("Epoch start: %d ------------" % (epoch))
    
    for i in range(0, len(ID_list)): #select one label of digits
        
        for j in range(0, len(ID_list[i])): #select the specific writing of that label
            
            optimizer.zero_grad()
            
            rho_ = image[i][j]
            ID = ID_list[i][j]
            label = get_mnist_label(ID)
            print(rho_)
            u = model(INPUTS, rho_).sum(dim=1)
            
            spintorch.plot.plot_output(u[0,], OUTPUTS[i]+1, epoch, label, ID, plotdir)
            loss = my_loss(u,OUTPUTS[i:i+1])
            loss_batch += loss.item()
            
            print('ID',ID)
            print('label',label)
            print('loss',loss.item())
            print('loss_batch',loss_batch)
            
            stat_cuda('after forward')
            
            if loss.item() >= loss_T:
                loss.backward()
                optimizer.step()
                
            stat_cuda('after backward')
            toc()
            
            if epoch >= epoch_init + epoch_number -1:
                spintorch.plot.geometry_multi(model, epoch=epoch, plotdir=plotdir, label=label, ID = ID)
                if model.retain_history:
                    with torch.no_grad():
                        mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
                        #wave_snapshot(model, mz[timesteps-1], (plotdir+'snapshot_time%d_epoch%d.png' % (timesteps,epoch)),r"$m_z$")
                        #wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+'snapshot_time%d_epoch%d.png' % (int(timesteps/2),epoch)),r"$m_z$")
                        wave_integrated(model, mz, (plotdir+'integrated_epoch%d_L%d_ID%d.png' % (epoch, label, ID)))
                        

                       
    
    print("Epoch finished: %d -- Loss: %.6f" % (epoch, loss_batch))
    loss_iter.append(loss_batch)  # store loss values
    spintorch.plot.plot_loss(loss_iter, plotdir)

    '''Save model checkpoint'''
    if epoch >= epoch_init + epoch_number -1:
        torch.save({
                'epoch': epoch,
                'loss_iter': loss_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savedir + 'model_e%d.pt' % (epoch))

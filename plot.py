import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.colors import CenteredNorm
from matplotlib.ticker import MaxNLocator
from .geom import WaveGeometryMs, WaveGeometry
from .solver import MMSolver
from .binarize import binarize

import warnings
warnings.filterwarnings("ignore", message=".*No contour levels were found.*")


mpl.use('Agg',) # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [16.0, 12.0]
mpl.rcParams['figure.dpi'] = 600


def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'loss.png')
    plt.close(fig)
    
def plot_output(u, p, epoch, label, ID, plotdir):
    fig = plt.figure()
    plt.bar(range(1,1+u.size()[0]), u.detach().cpu().squeeze(), color='k')
    plt.xlabel("output number")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'output_epoch%d_X%d_L%d_ID%d.png' % (epoch, p, label, ID))
    plt.close(fig)


def _plot_probes(probes, ax):
    markers = []
    for i, probe in enumerate(probes):
        x,y = probe.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='k',markersize=3,alpha=0.8)
        markers.append(marker)
    return markers


def _plot_sources(sources, ax):
    markers = []
    for i, source in enumerate(sources):
        x,y = source.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='g',markersize=3,alpha=0.8)
        markers.append(marker)
    return markers


def geometry(model, ax=None, outline=False, outline_pml=True, epoch=0, plotdir=''):

    geom = model.geom
    probes = model.probes
    sources = model.sources
    A = model.Alpha()[0, 0, ].squeeze()
    alph = A.min().cpu().numpy()
    B = geom.B[1,].detach().cpu().numpy().transpose()

    if ax is None:
        fig, ax = plt.subplots(1, 1) #constrained_layout=True
        
    markers = []
    if not outline:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Saturation magnetization (A/m)')
        else:
            h1 = ax.imshow(B*1e3, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, label='Magnetic field (mT)')
    else:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            ax.contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
        else:
            ax.contour(B, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

    markers += _plot_probes(probes, ax)
    markers += _plot_sources(sources, ax)
        
    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch))
        plt.close(fig)

def geometry_multi(model,rho, ax=None, outline=False, outline_pml=True, epoch=0, plotdir='', label=0, ID = 0):

    geom = model.geom
    probes = model.probes
    sources = model.sources
    A = model.Alpha()[0, 0, ].squeeze()
    alph = A.min().cpu().numpy()
    B1 = geom.B[1,].detach().cpu().numpy().transpose()
    B0 = geom.B[0,].detach().cpu().numpy().transpose()
    m_rho = geom.m_rho.detach().cpu().numpy().transpose()
    
    
    '''My code ttf19'''
    rho1_train = geom.rho1_train
    rho2_train = geom.rho2_train
    rx_train = geom.rx_train
    ry_train = geom.ry_train
    rx = geom.rx
    ry = geom.ry
    r0_train = geom.r0_train
    dr_train = geom.dr_train
    dr_input = geom.dr_input
    wm = geom.wm
    lm = geom.lm
    r0 = geom.r0

    # Angle of magnetisation direction in x-y plane. Units of pi
    rad_train = [1./2 - binarize(rho1_train).detach().cpu().numpy()/2,1. - binarize(rho2_train).detach().cpu().numpy()/2]
    rad_input = [1./2-binarize(rho[0]).detach().cpu().numpy()/2.,1.-binarize(rho[1]).detach().cpu().numpy()/2]

    # x and y positions of magnets for train then input. 1: magnets oriented in x direction. 2: oriented in y direction.
    xs1_train = np.arange(r0_train-int(dr_train/2),r0_train+rx_train*dr_train+int(dr_train/2)+0.5,dr_train)
    ys1_train = np.arange(r0,r0+ry_train*dr_train + 0.5,dr_train)

    xs1_input = np.arange(r0-int(dr_input/2),r0+rx*dr_input+int(dr_input/2)+0.5,dr_input) 
    ys1_input = np.arange(r0,r0+ry*dr_input+0.5,dr_input)
    
    xs2_train = np.arange(r0_train,r0_train+rx_train*dr_train+0.5,dr_train)
    ys2_train = np.arange(r0-int(dr_train/2),r0+ry_train*dr_train+int(dr_train/2)+0.5,dr_train)

    xs2_input = np.arange(r0,r0+rx*dr_input,dr_input)
    ys2_input = np.arange(r0-int(dr_input/2),r0+ry*dr_input+int(dr_input/2)+0.5,dr_input)

    cmap = mpl.cm.get_cmap('hsv')
    norm = mcolors.Normalize(vmin= 0,vmax = 2)
    #end


    if ax is None:
        fig, ax = plt.subplots(1, 1) #constrained_layout=True Commented out because threw error
        
    markers = []
    if not outline:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            h1 = ax.imshow(Msat, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax, fraction=0.02, pad=0.04,label='Saturation magnetization (A/m)')
        else:
            h1 = ax.imshow(B1*1e3, origin="lower", cmap=plt.cm.summer)
            plt.colorbar(h1, ax=ax,fraction=0.02, pad=0.04,label='Magnetic field (mT)')
    else:
        if isinstance(model.geom, WaveGeometryMs):
            Msat = geom.Msat.detach().cpu().numpy().transpose()
            ax.contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
        else:
            ax.contour(B1, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

    markers += _plot_probes(probes, ax)
    markers += _plot_sources(sources, ax)


    if plotdir:
        fig.savefig(plotdir+'geometry_epoch%d_L%d_ID%d.png' % (epoch , label, ID))
        plt.close(fig)

    '''
    #Display magnets and their magnetisation direction
    for x in range(rx_train):
        for y in range(ry_train):
            rect = patches.Rectangle((xs1_train[x]-lm/2, ys1_train[y]-wm/2), lm, wm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_train[0][x][y])))
            ax.add_patch(rect)
    for x in range(rx_train-1):
        for y in range(ry_train+1):
            rect = patches.Rectangle((xs2_train[x]-wm/2, ys2_train[y]-lm/2), wm, lm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_train[1][x][y])))
            ax.add_patch(rect)

    for x in range(rx+1):
        for y in range(ry):
            rect = patches.Rectangle((xs1_input[x]-lm/2, ys1_input[y]-wm/2), lm, wm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_input[0][x][y])))
            ax.add_patch(rect)
    for x in range(rx):
        for y in range(ry+1):
            rect = patches.Rectangle((xs2_input[x]-wm/2, ys2_input[y]-lm/2), wm, lm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_input[1][x][y])))
            ax.add_patch(rect)

    
    ''' #Uncomment this code to mark metastable high energy states in input
    x_HES_input = [] #HES = high energy state
    y_HES_input = []
    for x in range(rx):
        for y in range(ry):
            if rad_input[0][x][y]-rad_input[0][x+1][y]!=0 and rad_input[1][x][y]-rad_input[1][x][y+1]!=0 and rad_input[0][x][y]-rad_input[1][x][y]==-0.5:
                x_HES_input.append(x)
                y_HES_input.append(y)
    y_HES = r0 + np.array(y_HES_input)*dr_input
    x_HES = r0 + np.array(x_HES_input)*dr_input
    ax.plot(x_HES,y_HES,'o',color='k',markersize=5)
    '''
    
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,fraction=0.02, pad=0.04,label='Magnetisation angle')

    if plotdir:
        fig.savefig(plotdir+'mag_geometry_epoch%d_L%d_ID%d.png' % (epoch , label, ID))
        plt.close(fig)
    '''

    # Arrow plot
    fig, ax = plt.subplots()
    markers = []

    if outline_pml:
        b_boundary = A.cpu().numpy().transpose()
        ax.contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

    markers += _plot_probes(probes, ax)
    markers += _plot_sources(sources, ax)

    #Display magnets and their magnetisation direction
    for x in range(rx_train):
        for y in range(ry_train):
            rect = patches.Rectangle((xs1_train[x]-lm/2, ys1_train[y]-wm/2), lm, wm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_train[0][x][y])))
            ax.add_patch(rect)
    for x in range(rx_train-1):
        for y in range(ry_train+1):
            rect = patches.Rectangle((xs2_train[x]-wm/2, ys2_train[y]-lm/2), wm, lm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_train[1][x][y])))
            ax.add_patch(rect)

    for x in range(rx+1):
        for y in range(ry):
            rect = patches.Rectangle((xs1_input[x]-lm/2, ys1_input[y]-wm/2), lm, wm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_input[0][x][y])))
            ax.add_patch(rect)
    for x in range(rx):
        for y in range(ry+1):
            rect = patches.Rectangle((xs2_input[x]-wm/2, ys2_input[y]-lm/2), wm, lm, linewidth=1, edgecolor='k', facecolor=cmap(norm(rad_input[1][x][y])))
            ax.add_patch(rect)

    ax.quiver(B0*1e3,B1*1e3-60,minlength=0)
    plt.gca().set_aspect('equal', adjustable='box')

    if plotdir:
        fig.savefig(plotdir+'quiver_epoch%d_L%d_ID%d.png' % (epoch , label, ID))
        plt.close(fig)

    #fig, ax = plt.subplots(ncols=3)
    #print("plot",shape(m_rho))
    #ax[0].pcolor(m_rho[0][0])
    #ax[1].pcolor(m_rho[0][1])
    #ax[2].pcolor(m_rho[0][2])
    #plt.gca().set_aspect('equal', adjustable='box')
    if plotdir:
        fig.savefig(plotdir+'m_rho1_epoch%d_L%d_ID%d.png' % (epoch , label, ID))
        plt.close(fig)
    


        
def wave_integrated(model, m_history, filename=''):
    
    m_int = m_history.pow(2).sum(dim=0).numpy().transpose()
    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    vmax = m_int.max()
    h = ax.imshow(m_int, cmap=plt.cm.viridis, origin="lower", norm=LogNorm(vmin=vmax*0.01,vmax=vmax))
    plt.colorbar(h,fraction=0.046, pad=0.04)
    geometry(model, ax=ax, outline=True)

    if filename:
        fig.savefig(filename)
        plt.close(fig)


def wave_snapshot(model, m_snap, filename='', clabel='m'):
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    m_t = m_snap.cpu().numpy().transpose()
    h = axs.imshow(m_t, cmap=plt.cm.RdBu_r, origin="lower", norm=CenteredNorm())
    geometry(model, ax=axs, outline=True)
    plt.colorbar(h, ax=axs, label=clabel, shrink=0.80)
    axs.axis('image')
    if filename:
        fig.savefig(filename)
        plt.close(fig)
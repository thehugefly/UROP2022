"""Modules for representing the trained parameters"""

"""This file (geom.py) defines the geometry of the magnets. 
The forward function of each class outputs the magnetic field caused by these magnets"""

import torch
from torch import nn, sum, tensor, zeros, ones, real
from torch.fft import fftn, ifftn
from .demag import Demag
from .binarize import binarize
from numpy import pi
from scipy import io
import numpy as np



class WaveGeometry(nn.Module):
    def __init__(self, dim: tuple, d: tuple, B0: float, Ms_sheet: float):
        super().__init__()

        self.dim = dim
        self.d   = d
        self.register_buffer("B0", tensor(B0))
        self.register_buffer("Ms_sheet", tensor(Ms_sheet))


    def forward(self): 
        raise NotImplementedError


class WaveGeometryFreeForm(WaveGeometry):
    def __init__(self, dim: tuple, d: tuple, B0: float, B1: float, Ms: float):

        super().__init__(dim, d, B0, Ms)

        self.rho = nn.Parameter(zeros(dim))
        self.register_buffer("B", zeros((3,)+dim))
        self.register_buffer("B1", tensor(B1))
        self.B[1,] = self.B0
        
    def forward(self):
        self.B = torch.zeros_like(self.B)
        self.B[1,] = self.B1*self.rho + self.B0
        return self.B



class WaveGeometryMs(WaveGeometry):
    def __init__(self, dim: tuple, d: tuple, Ms: float, B0: float):

        super().__init__(dim, d, B0, Ms)

        self.rho = nn.Parameter(ones(dim))
        self.register_buffer("Msat", zeros(dim))
        self.register_buffer("B0", tensor(B0))
        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] = self.B0
        
    def forward(self):
        self.Msat = self.Ms*self.rho
        return self.Msat


class WaveGeometryArray(WaveGeometry):
    def __init__(self, rho, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int,
                  Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off
        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_CoPt", tensor(Ms_CoPt))
        self.rho = nn.Parameter(rho.clone().detach())
        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = ones((dm, dm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry = self.r0, self.dr, self.rx, self.ry
        rho_binary = binarize(self.rho)   
        m_rho = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary
        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]
        m_ = nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        
        self.B = B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]*self.Ms_CoPt*mu0
        self.B[1,] += self.B0
        return self.B

class WaveGeometryArray_draw(WaveGeometry):
    def __init__(self, rho, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int,
                  Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off
        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_CoPt", tensor(Ms_CoPt))
        self.rho = rho#nn.Parameter(rho.clone().detach())
        
        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = ones((dm, dm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry = self.r0, self.dr, self.rx, self.ry
        rho_binary = binarize(self.rho)  
        #print(rho_binary) 
        m_rho = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary
        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]
        #print(m_rho_)
        m_ = nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        #print(B_demag)
        self.B = B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]*self.Ms_CoPt*mu0
        self.B[1,] += self.B0
        return self.B




class WaveGeometryArray_draw_and_train_x_multi(WaveGeometry):
    def __init__(self, rho_train, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int, r0_train : int,rx_train : int,ry_train : int,
                  Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off

        self.r0_train = r0_train
        self.rx_train = rx_train
        self.ry_train = ry_train

        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_CoPt", tensor(Ms_CoPt))
        self.rho_train = nn.Parameter(rho_train.clone().detach())
        
        # self.rho1 = rho1
        # self.rho2 = rho2
        # self.rho = rho

        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = ones((dm, dm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self, rho):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry, r0_train, rx_train, ry_train = self.r0, self.dr, self.rx, self.ry, self.r0_train, self.rx_train, self.ry_train
        
        # rho1_binary = binarize(self.rho1) 
        # rho2_binary = binarize(self.rho2)
        # rho_binary = torch.cat((rho1_binary, rho2_binary))
        # print(self.rho.shape)
        # print(self.rho)
        # rho_binary = binarize(self.rho)
        rho_binary = rho
        # print(rho_binary.shape)
        # print(rho_binary)
        rho_train_binary = binarize(self.rho_train)
        m_rho = zeros((1, 3, ) + self.dim, device=self.B0.device)
        
#         for i in range(0, 2):
#             m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary[i:i+1]
        #print(rx)
        #print(ry)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary
        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]

        m_rho_train = zeros((1, 3, ) + self.dim, device=self.B0.device)
        #print(r0,rx,ry)
        #print(r0_train,rx_train,ry_train)
        m_rho_train[0, 2, r0_train:r0_train+rx_train*dr:dr, r0:r0+ry_train*dr:dr] = rho_train_binary
        m_rho_train_ = self.convolver(m_rho_train)[:,:,0:nx,0:ny]
        #print(m_rho_)
        #print(m_rho_)
        #print(m_rho_train_)
        #cv2.waitkey()
        m_all = torch.cat([m_rho_,m_rho_train_],axis=0)
        m_ = nn.functional.pad(m_rho.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_train_ = nn.functional.pad(m_rho_train_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        m_fft_train = fftn(m_train_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        B_demag_train = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft_train),1),
                                          sum((self.Ky_fft*m_fft_train),1),
                                          sum((self.Kz_fft*m_fft_train),1)], 1), dim=(2,3)))
        #print(B_demag)
        self.B = (B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]+B_demag_train[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0])*self.Ms_CoPt*mu0
        self.B[1,] += self.B0
        return self.B

class WaveGeometryArray_draw_and_train_x(WaveGeometry):
    def __init__(self, rho,rho_train, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, dm: int, z_off: int, rx: int, ry: int, r0_train : int,rx_train : int,ry_train : int,
                  Ms_CoPt: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.dm = dm
        self.z_off = z_off
        self.r0_train = r0_train
        self.rx_train = rx_train
        self.ry_train = ry_train
        #print(rho_train)
        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_CoPt", tensor(Ms_CoPt))
        self.rho_train = nn.Parameter(rho_train.clone().detach())
        self.rho = rho
        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        
        for i in range(3):
            self.convolver.weight[i, 0, ] = ones((dm, dm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry, r0_train, rx_train, ry_train = self.r0, self.dr, self.rx, self.ry, self.r0_train, self.rx_train, self.ry_train
        rho_binary = binarize(self.rho)  
        rho_train_binary = binarize(self.rho_train)
        m_rho = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho[0, 2, r0:r0+rx*dr:dr, r0:r0+ry*dr:dr] = rho_binary
        m_rho_ = self.convolver(m_rho)[:,:,0:nx,0:ny]

        m_rho_train = zeros((1, 3, ) + self.dim, device=self.B0.device)
        print(r0,rx,ry)
        print(r0_train,rx_train,ry_train)
        m_rho_train[0, 2, r0_train:r0_train+rx_train*dr:dr, r0:r0+ry_train*dr:dr] = rho_train_binary
        m_rho_train_ = self.convolver(m_rho_train)[:,:,0:nx,0:ny]
        #print(m_rho_)
        print(m_rho_)
        print(m_rho_train_)
        #cv2.waitkey()
        m_all = torch.cat([m_rho_,m_rho_train_],axis=0)
        m_ = nn.functional.pad(m_rho.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_train_ = nn.functional.pad(m_rho_train_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        m_fft_train = fftn(m_train_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        B_demag_train = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft_train),1),
                                          sum((self.Ky_fft*m_fft_train),1),
                                          sum((self.Kz_fft*m_fft_train),1)], 1), dim=(2,3)))
        #print(B_demag)
        self.B = (B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]+B_demag_train[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0])*self.Ms_CoPt*mu0
        self.B[1,] += self.B0
        return self.B


class WaveGeometrySpinIce(WaveGeometry):
    def __init__(self, rho1, rho2, dim: tuple, d: tuple, Ms: float, B0: float,
                  r0: int, dr: int, wm: int, lm: int, z_off: int, rx: int, ry: int,
                  Ms_magnet: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms)
        self.r0 = r0
        self.dr = dr
        self.rx = rx
        self.ry = ry
        self.wm = wm
        self.lm = lm
        self.z_off = z_off

        ''' My code ttf19
        self.r0_train = r0_train
        self.rx_train = rx_train
        self.ry_train = ry_train
        '''

        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_magnet", tensor(Ms_magnet))
        self.rho1 = nn.Parameter(rho1.clone().detach())
        self.rho2 = nn.Parameter(rho2.clone().detach())

        self.convolver1 = nn.Conv2d(3, 3, (self.wm, self.lm), padding=(self.wm//2,self.lm//2),
                                    groups=3, bias=False)
        self.convolver1.weight.requires_grad = False
        self.convolver2 = nn.Conv2d(3, 3, (self.lm, self.wm), padding=(self.lm//2,self.wm//2),
                                    groups=3, bias=False)
        self.convolver2.weight.requires_grad = False

        
        for i in range(3):
            self.convolver1.weight[i, 0, ] = ones((wm, lm))
            self.convolver2.weight[i, 0, ] = ones((lm, wm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self,rho):
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr, rx, ry = self.r0, self.dr, self.rx, self.ry

        ''' My code ttf19
        r0_train,rx_train,ry_train = self.r0_train,self.rx_train,self.ry_train'''

        rho_binary = rho

        rho1_binary = binarize(self.rho1)   
        rho2_binary = binarize(self.rho2)   
        m_rho1 = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho2 = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho1[0, 1, r0-int(dr/2):r0+rx*dr-int(dr/2):dr, r0:r0+ry*dr:dr] = rho1_binary
        m_rho2[0, 0, r0:r0+rx*dr-dr:dr, r0-int(dr/2):r0+ry*dr+int(dr/2):dr] = rho2_binary
        m_rho_ = self.convolver1(m_rho1)[:,:,0:nx,0:ny]
        m_rho_ += self.convolver2(m_rho2)[:,:,0:nx,0:ny]
        m_ = nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        
        self.B = B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]*self.Ms_magnet*mu0
        self.B[1,] += self.B0
        print("B",self.B)
        # save variables to plot in Matlab:
        io.savemat('magnets.mat', dict(m = m_rho_.detach().cpu().numpy(),B = self.B.detach().cpu().numpy()))
        return self.B



class WaveGeometrySpinIce2(WaveGeometry):
    '''Define the device geometry in the case of square spin ice input and training array'''
    def __init__(self, rho1_train, rho2_train, dim: tuple, d: tuple, Ms_sheet: float, B0: float,
                  r0: int, dr_input: int, dr_train: int, wm: int, lm: int, z_off: int, rx: int, ry: int, r0_train : int, rx_train : int,ry_train : int,
                  Ms_magnet: float, beta: float = 100.0):

        super().__init__(dim, d, B0, Ms_sheet)
        self.r0 = r0
        self.dr_input = dr_input
        self.dr_train = dr_train
        self.rx = rx
        self.ry = ry
        self.wm = wm
        self.lm = lm
        self.z_off = z_off

        self.r0_train = r0_train
        self.rx_train = rx_train
        self.ry_train = ry_train

        self.register_buffer("beta", tensor(beta))
        self.register_buffer("Ms_magnet", tensor(Ms_magnet))
        self.rho1_train = nn.Parameter(rho1_train.clone().detach())
        self.rho2_train = nn.Parameter(rho2_train.clone().detach())

        #Note: convolver1 used to convolve x direction magnets. colvolver2 for y direction
        self.convolver2 = nn.Conv2d(3, 3, (self.wm, self.lm), padding=(self.wm//2,self.lm//2),
                                    groups=3, bias=False)
        self.convolver2.weight.requires_grad = False
        self.convolver1 = nn.Conv2d(3, 3, (self.lm, self.wm), padding=(self.lm//2,self.wm//2),
                                    groups=3, bias=False)
        self.convolver1.weight.requires_grad = False

        ''' #Old convolver for up/down magnets
        # MUST REMOVE!
        self.convolver = nn.Conv2d(3, 3, self.dm, padding=(self.dm//2),
                                    groups=3, bias=False)
        self.convolver.weight.requires_grad = False
        '''

        for i in range(3):
            self.convolver2.weight[i, 0, ] = ones((wm, lm))
            self.convolver1.weight[i, 0, ] = ones((lm, wm))
        
        self.demag_nanomagnet = Demag(self.dim, self.d)
        Kx_fft, Ky_fft, Kz_fft = self.demag_nanomagnet.demag_tensor_fft(int(self.z_off))
        self.register_buffer("Kx_fft", Kx_fft)
        self.register_buffer("Ky_fft", Ky_fft)
        self.register_buffer("Kz_fft", Kz_fft)

        self.register_buffer("B", zeros((3,)+dim))
        self.B[1,] += self.B0

    def forward(self,rho):
        #Note: rho is input mnist image and rho_train is trainable nanomagnet array
        mu0 = 4*pi*1e-7
        nx, ny, nz = int(self.dim[0]), int(self.dim[1]), 1
        r0, dr_input, dr_train, rx, ry = self.r0, self.dr_input, self.dr_train, self.rx, self.ry

        r0_train,rx_train,ry_train = self.r0_train,self.rx_train,self.ry_train

        rho1_train_binary = binarize(self.rho1_train)   
        rho2_train_binary = binarize(self.rho2_train)
        rho1_binary = binarize(rho[0])
        rho2_binary = binarize(rho[1])

        #Placing magnetisation of each nanomagnet in right position and convolving both x and y direction magnets into m_rho
        m_rho1 = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho2 = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho1[0, 0, r0-int(dr_input/2):r0+rx*dr_input+int(dr_input/2):dr_input, r0:r0+ry*dr_input:dr_input] = rho1_binary
        m_rho2[0, 1, r0:r0+rx*dr_input:dr_input, r0-int(dr_input/2):r0+ry*dr_input+int(dr_input/2):dr_input] = rho2_binary
        m_rho_ = self.convolver1(m_rho1)[:,:,0:nx,0:ny]
        m_rho_ += self.convolver2(m_rho2)[:,:,0:nx,0:ny]
        self.m_rho = m_rho_.clone()
        #print(self.m_rho.size())

        m_rho1_train = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho2_train = zeros((1, 3, ) + self.dim, device=self.B0.device)
        m_rho1_train[0, 0, r0_train-int(dr_train/2):r0_train+rx_train*dr_train+int(dr_train/2):dr_train, r0:r0+ry_train*dr_train:dr_train] = rho1_train_binary
        m_rho2_train[0, 1, r0_train:r0_train+rx_train*dr_train:dr_train, r0-int(dr_train/2):r0+ry_train*dr_train+int(dr_train/2):dr_train] = rho2_train_binary
        m_rho_train_ = self.convolver1(m_rho1_train)[:,:,0:nx,0:ny]
        m_rho_train_ += self.convolver2(m_rho2_train)[:,:,0:nx,0:ny]

        m_ = nn.functional.pad(m_rho_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft = fftn(m_, dim=(2,3))
        B_demag = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft),1),
                                          sum((self.Ky_fft*m_fft),1),
                                          sum((self.Kz_fft*m_fft),1)], 1), dim=(2,3)))
        
        
        m_train_ = nn.functional.pad(m_rho_train_.unsqueeze(4), (0, nz, 0, ny, 0, nx))  
        m_fft_train = fftn(m_train_, dim=(2,3))
        B_demag_train = real(ifftn(torch.stack([sum((self.Kx_fft*m_fft_train),1),
                                          sum((self.Ky_fft*m_fft_train),1),
                                          sum((self.Kz_fft*m_fft_train),1)], 1), dim=(2,3)))
        #end
        
        #Should this line have Ms_magnet or Ms_sheet?
        self.B = (B_demag[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0]+B_demag_train[0,:,nx-1:2*nx-1,ny-1:2*ny-1,0])*self.Ms_magnet*mu0 #edited line to add train component
        self.B[1,] += self.B0
        return self.B
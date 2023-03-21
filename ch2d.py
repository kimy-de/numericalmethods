"""
Heat equation
∂u/∂t - α ∇²u = 0
"""

from __future__ import print_function
import os
from fenics import *
import time
import matplotlib.pyplot as plt
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

class CH(nn.Module):
    def __init__(self, dt, c, r, h):
        super(CH, self).__init__()

        self.alpha1 = c / (h ** 2)
        self.alpha2 = dt * c / (h ** 2)
        self.beta = r
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1) # Zero Neumann boundary condition

    def forward(self, x):
        u_pad = self.pad(x)
        z = F.conv2d(u_pad, self.delta)
        x1 = -(1/self.beta)*self.alpha1 * z - x + x**3
        x_pad = self.pad(x1)
        x2 = x + self.alpha2*F.conv2d(x_pad, self.delta)
        return x2

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # domain setting
    nx = 100 # nx=ny
    h = 1/nx # dx=dy
    dt = 0.01 * h ** 2
    max_time = 0.005
    max_iter = int(max_time/dt)
    print("Number of iteraions:", max_iter)

    x = torch.linspace(0, 1, nx)  # x domain [0,1]
    y = torch.linspace(0, 1, nx)  # y domain [0,1]

    # Cahn-Hilliard equation with the zero Neumann boundary condition
    r = 6944
    c = 1
    fdm = CH(dt, c, r, h) 

    # initial condition
    u = (2 * torch.rand(nx,nx) - 1).view(1,1,nx,nx).to(device)

    # time evolution
    plt.figure(figsize=(5, 5)) 
    z = 0
    with torch.no_grad(): 
        for i in range(max_iter):
            u = fdm(u) 
            if (i % 50 == 0) and (i>=500):
                plt.imshow(u.view(nx,nx), interpolation='nearest', cmap='copper',
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower', aspect='auto')
                plt.clim(-1, 1)
                plt.xticks([0,1])
                plt.yticks([0,1])
                plt.savefig('./time'+str(z)+'.jpg')
                z += 1

    with imageio.get_writer('./ch2d.gif', mode='I') as writer:
        for i in range(z):
            image = imageio.v2.imread('./time'+str(i)+'.jpg')
            writer.append_data(image)

            os.remove('./time'+str(i)+'.jpg')
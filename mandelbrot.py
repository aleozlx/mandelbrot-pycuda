from __future__ import print_function
from __future__ import division
from time import time
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

def mandelbrot_GPU(nx, ny):
    linear = (nx * ny,)
    x, y = np.ogrid[-2:1:nx*1j, -1.5:1.5:ny*1j]
    z = (x + 1j*y).astype(np.complex64).reshape(linear)
    mandelbrot = lambda N: SourceModule("""
    #include <pycuda-complex.hpp>
    __global__ void mandelbrot(pycuda::complex<float> *z, pycuda::complex<float> *c, int *r)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if( i < %d ){
            r[i] = -1;
            for(int j=0; j<80; ++j)
                if (abs(z[i]) <= 4) z[i] = z[i]*z[i] + c[i];
                else { r[i] = j; break; }
        }
    } """ % N[0])
    ret = gpuarray.empty(linear, np.int32)
    kernel = mandelbrot(linear).get_function("mandelbrot")
    t_start = time()
    kernel(gpuarray.to_gpu(z), gpuarray.to_gpu(z), ret, grid=(((linear[0]+1)//1024),1,1), block=(1024,1,1))
    return ret.get().reshape((nx, ny)), time()-t_start

r1 = mandelbrot_GPU(800, 800)

def mandelbrot_CPU(nx, ny):
    x, y = np.ogrid[-2:1:nx*1j, -1.5:1.5:ny*1j]
    c = x + 1j*y; z = c.copy()
    mask = np.ones(z.shape, dtype=bool)
    mandelbrot = -np.ones(z.shape)
    t_start = time()
    for ct in range(80):
        contained = abs(z) <= 4.0
        changed = np.logical_xor(mask, contained)
        mandelbrot[changed] = ct
        # yield mandelbrot.copy()
        mask &= contained
        z[contained] = z[contained]**2 + c[contained]
    return mandelbrot, time()-t_start

r2 = mandelbrot_CPU(800, 800)

print('err:', round(np.linalg.norm(r1[0] - r2[0]),3), 'speed up:', round(r2[1]/r1[1],2))
np.save('r1.npy', r1[0], allow_pickle=False)
np.save('r2.npy', r2[0], allow_pickle=False)
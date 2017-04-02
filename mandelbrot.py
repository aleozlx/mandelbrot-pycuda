import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.curandom import rand as curand
from pycuda.elementwise import ElementwiseKernel

mandelbrot = """
    for(int j=0; j<80; ++j)
        if (abs(z[i]) <= 4) z[i] = z[i]*z[i] + c[i];
        else { r[i] = j; break; }
"""

def mandelbrot_GPU(nx, ny):
    linear = (nx * ny,)
    x, y = np.ogrid[-2:1:nx*1j, -1.5:1.5:ny*1j]
    z = (x + 1j*y).reshape(linear)
    
    ret = gpuarray.empty(linear, np.int32)
    kernel = ElementwiseKernel("pycuda::complex<float> *z, pycuda::complex<float> *c, int *r",
        mandelbrot, "mandelbrot", preamble = "#include <pycuda-complex.hpp>")
    kernel(gpuarray.to_gpu(z), gpuarray.to_gpu(z), ret)
    return ret.get().reshape((nx, ny))

r1 = mandelbrot_GPU(800, 800)
print r1

def mandelbrot_CPU(nx, ny):
    x, y = np.ogrid[-2:1:nx*1j, -1.5:1.5:ny*1j]
    c = x + 1j*y; z = c.copy()
    mask = np.ones(z.shape, dtype=bool)
    mandelbrot = -np.ones(z.shape)#*60
    for ct in range(80):
        contained = abs(z)<=4 # those still contained
        changed = np.logical_xor(mask, contained)
        mandelbrot[changed] = ct
        yield mandelbrot.copy()
        mask &= contained
        z[contained] = z[contained]**2 + c[contained]

r2 = list(mandelbrot_CPU(800, 800))[-1]
print r2

print np.linalg.norm(r1 - r2)
np.save('r1.npy', r1, allow_pickle=False)
np.save('r2.npy', r2, allow_pickle=False)
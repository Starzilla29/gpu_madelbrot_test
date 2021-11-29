from matplotlib import pyplot as plt
import numpy as np
import math
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8
from pylab import imshow, show
from timeit import default_timer as timer

########################################
########### CPU FUNCTIONS ##############
########################################
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters

def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

def generate_mandelbrot_with_cpu():
    image = np.zeros((1024, 1536), dtype = np.uint8)
    start = timer()
    create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
    dt = timer() - start

    print("Mandelbrot created in %f s" % dt)
    imshow(image)
    show()

########################################
########### JIT FUNCTIONS ##############
########################################
@jit
def mandel_jit(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the Mandelbrot
    set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i
    return max_iters

@jit
def create_fractal_jit(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel_jit(real, imag, iters)
            image[y, x] = color

def generate_mandelbrot_with_numba():
    """
    takes our python code and turns it into machine code with the JIT compiler making it have a potential to be
    much much faster.
    """
    image = np.zeros((1024, 1536), dtype=np.uint8)
    start = timer()
    create_fractal_jit(-2.0, 1.0, -1.0, 1.0, image, 20)
    dt = timer() - start

    print("Mandelbrot created in %f s" % dt)
    imshow(image)
    show()

########################################
########### GPU FUNCTIONS ##############
########################################

# mandel_gpu = cuda.jit(uint32(f8, f8, uint32), device = True)(mandel)

# def mandel(x, y, max_iters):
#     """
#     Given the real and imaginary parts of a complex number,
#     determine if it is a candidate for membership in the Mandelbrot
#     set given a fixed number of iterations.
#     """
#     c = complex(x, y)
#     z = 0.0j
#     for i in range(max_iters):
#         z = z*z + c
#         if (z.real*z.real + z.imag*z.imag) >= 4:
#             return i
#     return max_iters

# @cuda.jit(argtypes=[f8, f8, f8, f8, uint8[:,:], uint32])
# def create_fractal_gpu(min_x, max_x, min_y, max_y, image, iters):
#     height = image.shape[0]
#     width = image.shape[1]
#
#     pixel_size_x = (max_x - min_x) / width
#     pixel_size_y = (max_y - min_y) / height
#
#     startX, startY = cuda.grid(2)
#     gridX = cuda.gridDim.x * cuda.blockDim.x;
#     gridY = cuda.gridDim.y * cuda.blockDim.y;
#
#     for x in range(startX, width, gridX):
#         real = min_x + x * pixel_size_x
#         for y in range(startY, height, gridY):
#             imag = min_y + y * pixel_size_y
#             color = mandel_gpu(real, imag, iters)
#             image[y, x] = color
#
# def generate_mandelbrot_with_gpu():
#     image = np.zeros((1024, 1536), dtype=np.uint8)
#     start = timer()
#     create_fractal_gpu(-2.0, 1.0, -1.0, 1.0, image, 20)
#     dt = timer() - start
#
#     print("Mandelbrot created in %f s" % dt)
#     imshow(image)
#     show()

if __name__ == '__main__':
    generate_mandelbrot_with_cpu()

nvidia-docker run -it  -v "$(readlink -f .)":/workspace nightseas/pycuda bash #python /workspace/mandelbrot.py

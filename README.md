# cudaocc

A `ctypes`-based interface to `cuda_occupancy.h`


## Installation

I recommend the use of virtual environment:

```
python3 -m venv .venv && source .venv/bin/activate
```

First, build and install the C libraries. Make sure `nvcc` is in the
path.

```
make CUDA_INSTALL_PATH=/path/to/cuda
```

Then, assuming you are in a virtual environment:

```
make install
```

will copy `gen_device_prop` to the binary directory. It will also display a `LD_LIBRARY_PATH` setting. Set that as follows (replace `...` with the directory reported by `make install`:

```
export LD_LIBRARY_PATH=...
```

Then, build and install the Python module:

```
pip install .
```

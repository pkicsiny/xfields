import numpy as np

from .base import XfBasePlatform

try:
    import cupy
    from cupyx.scipy import fftpack as cufftp
except ImportError:
    print('WARNING: cupy is not installed, this platform will not be available')
    from .platnotavail import ModuleNotAvailable
    cupy = ModuleNotAvailable(message=('cupy is not installed. '
                            'this platform is not available!'))
    cufftp = cupy

from .default_kernels import cupy_default_kernels

class MinimalDotDict(dict):
    def __getattr__(self, attr):
        return self.get(attr)

class XfCupyPlatform(XfBasePlatform):

    """Creates a Cupy Platform object, that allows performing the
    computations     on nVidia GPUs.

    Args:
        default_kernels (bool): If ``True``, the Xfields defult kernels are
            automatically imported.
        default_block_size (int):  CUDA thread size that is used by default
            for kernel execution in case a block size is not specified
            directly in the kernel object. The default value is 256.
    Returns:
        XfCupyPlatform: platform object.

    """

    def __init__(self, default_kernels=True, default_block_size=256):

        self.default_block_size = default_block_size
        self.kernels = MinimalDotDict()

        if default_kernels:
            self.add_kernels(src_files=cupy_default_kernels['src_files'],
                    kernel_descriptions=cupy_default_kernels['kernel_descriptions'])

    @property
    def nplike_lib(self):
        """
        Module containing all the numpy features supported by cupy.

        Example:

        .. code-block:: python

            platform =  XfCupyPlatform()
            nplike = platform.nplike_lib

            # This returns an array of zeros on the computing device (GPU):
            a = nplike.zeros((10,10), dtype=nplike.float64

        """
        return cupy

    def nparray_to_platform_mem(self, arr):
        """Copies a numpy array to the device memory.
        Args:
            arr (numpy.ndarray): Array to be transferred

        Returns:
            cupy.ndarray:The same array copied to the device.

        """
        dev_arr = cupy.array(arr)
        return dev_arr

    def nparray_from_platform_mem(self, dev_arr):
        """Copies an array to the device to a numpy array.

        Args:
            dev_arr (cupy.ndarray): Array to be transferred/
        Returns:
            numpy.ndarray: The same data copied to a numpy array.

        """
        return dev_arr.get()

    def plan_FFT(self, data, axes, ):
        """Generate an FFT plan object to be executed on the platform.

        Args:
            data (cupy.ndarray): Array having type and shape for which the FFT
                needs to be planned.
            axes (sequence of ints): Axes along which the FFT needs to be
                performed.
        Returns:
            XfCupyFFT: FFT plan for the required array shape, type and axes.

        Example:

        .. code-block:: python

            plan = platform.plan_FFT(data, axes=(0,1))

            data2 = 2*data

            # Forward tranform (in place)
            plan.transform(data2)

            # Inverse tranform (in place)
            plan.itransform(data2)
        """
        return XfCupyFFT(self, data, axes)

    def add_kernels(self, src_code='', src_files=[], kernel_descriptions={}):

        """Adds user-defined kernels to to the platform. The kernel source
           code is provided as a string and/or in source files and must contain
           the kernel names defined in the kernel descriptions.

        Args:
            src_code (str): String with the kernel source code. Default: empty 
                string.
            src_files (list of strings): paths to files containing the
                source code. Default: empty list.
            kernel_descriptions (dict): Dictionary with the kernel descriptions
                in the form given by the following examples. The decriptions
                define the kernel names, the type and name of the arguments
                and identifies one input argument that defines the number of
                threads to be launched.

        Example:

        .. code-block:: python

            src_code = r'''
            __global__
            void my_mul(const int n, const float* x1,
                        const float* x2, float* y) {
                int tid = blockDim.x * blockIdx.x + threadIdx.x;
                if (tid < n){
                    y[tid] = x1[tid] * x2[tid];
                    }
                }
            '''
            kernel_descriptions = {'my_mul':{
                args':(
                    (('scalar', np.int32),   'n',),
                    (('array',  np.float64), 'x1',),
                    (('array',  np.float64), 'x2',),
                    )
                'num_threads_from_arg': 'nparticles'
                },}

            # Import kernel in platform
            platform.add_kernels(src_code, kernel_descriptions)

            # With a1 and a2 being arrays on the platform, the kernel
            # can be called as follows:
            platform.kernels.my_mul(n=len(a1), x1=a1, x2=a2)
        """

        src_content = 'extern "C"{'
        for ff in src_files:
            with open(ff, 'r') as fid:
                src_content += ('\n\n' + fid.read())
        src_content += "}"

        module = cupy.RawModule(code=src_content)

        ker_names = kernel_descriptions.keys()
        for nn in ker_names:
            kk = module.get_function(nn)
            aa = kernel_descriptions[nn]['args']
            nt_from = kernel_descriptions[nn]['num_threads_from_arg']
            aa_types, aa_names = zip(*aa)
            self.kernels[nn] = XfCupyKernel(cupy_kernel=kk,
                arg_names=aa_names, arg_types=aa_types,
                num_threads_from_arg=nt_from,
                block_size=self.default_block_size)

class XfCupyKernel(object):

    def __init__(self, cupy_kernel, arg_names, arg_types,
                 num_threads_from_arg, block_size):

        assert (len(arg_names) == len(arg_types))
        assert num_threads_from_arg in arg_names

        self.cupy_kernel = cupy_kernel
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.num_threads_from_arg = num_threads_from_arg
        self.block_size = block_size

    @property
    def num_args(self):
        return len(self.arg_names)

    def __call__(self, **kwargs):
        assert len(kwargs.keys()) == self.num_args
        arg_list = []
        for nn, tt in zip(self.arg_names, self.arg_types):
            vv = kwargs[nn]
            if tt[0] == 'scalar':
                assert np.isscalar(vv)
                arg_list.append(tt[1](vv))
            elif tt[0] == 'array':
                assert isinstance(vv, cupy.ndarray)
                arg_list.append(vv.data)
            else:
                raise ValueError(f'Type {tt} not recognized')

        n_threads = kwargs[self.num_threads_from_arg]
        grid_size = int(np.ceil(n_threads/self.block_size))
        self.cupy_kernel((grid_size, ), (self.block_size, ), arg_list)


class XfCupyFFT(object):
    def __init__(self, platform, data, axes):

        self.platform = platform
        self.axes = axes

        assert len(data.shape) > max(axes)

        from cupyx.scipy import fftpack as cufftp
        self._fftplan = cufftp.get_fft_plan(
                data, axes=self.axes, value_type='C2C')

    def transform(self, data):
        data[:] = cufftp.fftn(data, axes=self.axes, plan=self._fftplan)[:]
        """The transform is done inplace"""


    def itransform(self, data):
        """The transform is done inplace"""
        data[:] = cufftp.ifftn(data, axes=self.axes, plan=self._fftplan)[:]

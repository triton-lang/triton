import pyopencl as cl
import pyviennacl as vcl

all_devices = [d for platform in cl.get_platforms() for d in platform.get_devices()]

DEVICE_TYPE_PREFIX = {  cl.device_type.GPU: 'gpu',
                        cl.device_type.CPU: 'cpu',
                        cl.device_type.ACCELERATOR: 'accelerator'
}

DEVICE_TYPE_CL_NAME = { cl.device_type.GPU: 'CL_DEVICE_TYPE_GPU',
                        cl.device_type.CPU: 'CL_DEVICE_TYPE_CPU',
                        cl.device_type.ACCELERATOR: 'CL_DEVICE_TYPE_ACCELERATOR'
}

VENDOR_PREFIX = {       vcl.opencl.VendorId.beignet_id: 'beignet',
                        vcl.opencl.VendorId.nvidia_id: 'nvidia',
                        vcl.opencl.VendorId.amd_id: 'amd',
                        vcl.opencl.VendorId.intel_id: 'intel'
}

DEVICES_PRESETS = {'all': all_devices,
                   'gpus': [d for d in all_devices if d.type==cl.device_type.GPU],
                   'cpus': [d for d in all_devices if d.type==cl.device_type.CPU],
                   'accelerators': [d for d in all_devices if d.type==cl.device_type.ACCELERATOR]
}



def sanitize_string(string, keep_chars = ['_']):
    string = string.replace(' ', '_').lower()
    string = "".join(c for c in string if c.isalnum() or c in keep_chars).rstrip()
    return string

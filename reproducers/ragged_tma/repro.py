#!/usr/bin/env python3
"""Reproduce Blackwell ragged-TMA allocation-mode faults using only libcuda."""
import argparse
import ctypes as c
import json
from pathlib import Path
import subprocess
import sys


U64, U32, SIZE, PTR, INT = c.c_uint64, c.c_uint32, c.c_size_t, c.c_void_p, c.c_int


class Location(c.Structure):
    _fields_ = [('type', INT), ('id', INT)]


class AllocationProp(c.Structure):
    _fields_ = [('type', INT), ('requestedHandleTypes', INT), ('location', Location),
                ('win32HandleMetaData', PTR), ('allocFlags', c.c_ubyte * 8)]


class AccessDesc(c.Structure):
    _fields_ = [('location', Location), ('flags', INT)]


class CudaError(RuntimeError):
    def __init__(self, function, code, name):
        self.code = code
        super().__init__(f'{function}: {name} ({code})')


def emit(**record):
    print(json.dumps(record), flush=True)


def load_driver():
    driver = c.CDLL('libcuda.so.1')
    signatures = {
        'cuInit': [U32],
        'cuDeviceGet': [c.POINTER(INT), INT],
        'cuDeviceGetName': [PTR, INT, INT],
        'cuDeviceGetAttribute': [c.POINTER(INT), INT, INT],
        'cuDriverGetVersion': [c.POINTER(INT)],
        'cuCtxCreate_v2': [c.POINTER(PTR), U32, INT],
        'cuCtxDestroy_v2': [PTR],
        'cuCtxSynchronize': [],
        'cuGetErrorName': [INT, c.POINTER(c.c_char_p)],
        'cuMemGetAllocationGranularity': [c.POINTER(SIZE), c.POINTER(AllocationProp), INT],
        'cuMemAddressReserve': [c.POINTER(U64), SIZE, SIZE, U64, U64],
        'cuMemCreate': [c.POINTER(U64), SIZE, c.POINTER(AllocationProp), U64],
        'cuMemMap': [U64, SIZE, SIZE, U64, U64],
        'cuMemSetAccess': [U64, SIZE, c.POINTER(AccessDesc), SIZE],
        'cuMemAlloc_v2': [c.POINTER(U64), SIZE],
        'cuMemcpyHtoD_v2': [U64, PTR, SIZE],
        'cuMemcpyDtoH_v2': [PTR, U64, SIZE],
        'cuTensorMapEncodeTiled': [PTR, INT, U32, PTR, c.POINTER(U64), c.POINTER(U64),
                                   c.POINTER(U32), c.POINTER(U32), INT, INT, INT, INT],
        'cuModuleLoadDataEx': [c.POINTER(PTR), PTR, U32, c.POINTER(INT), c.POINTER(PTR)],
        'cuModuleGetFunction': [c.POINTER(PTR), PTR, c.c_char_p],
        'cuLaunchKernel': [PTR, U32, U32, U32, U32, U32, U32, U32, PTR, c.POINTER(PTR), PTR],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes = arguments
        function.restype = INT
    return driver


def run_case(case, ordinal):
    stage, effective_class = 'setup', None
    try:
        driver = load_driver()

        def call(name, *args):
            code = getattr(driver, name)(*args)
            if code:
                label = c.c_char_p()
                driver.cuGetErrorName(code, c.byref(label))
                raise CudaError(name, code, (label.value or b'UNKNOWN').decode())

        call('cuInit', 0)
        device, major, minor, version = INT(), INT(), INT(), INT()
        call('cuDeviceGet', c.byref(device), ordinal)
        call('cuDeviceGetAttribute', c.byref(major), 75, device)
        call('cuDeviceGetAttribute', c.byref(minor), 76, device)
        call('cuDriverGetVersion', c.byref(version))
        name = c.create_string_buffer(256)
        call('cuDeviceGetName', name, len(name), device)
        libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if '/libcuda.so' in line})
        emit(kind='environment', device=name.value.decode(), compute_capability=[major.value, minor.value],
             driver_api=version.value, libcuda=libraries, case=case)
        if major.value not in (10, 11, 12):
            raise RuntimeError('This diagnostic edits a Blackwell-specific descriptor bit; use SM 10.x, 11.x or 12.x.')
        supported = INT()
        call('cuDeviceGetAttribute', c.byref(supported), 102, device)
        if not supported.value:
            raise RuntimeError('CUDA virtual memory management is required for the unmapped guard.')
        context = PTR()
        call('cuCtxCreate_v2', c.byref(context), 0, device)
        try:
            # A generic sm_90 PTX target permits driver JIT for the actual GPU.
            # No architecture-specific cubin, ptxas executable or Triton build is needed.
            ptx = c.create_string_buffer(Path(__file__).with_name('load.ptx').read_bytes())
            module = PTR()
            error_log = c.create_string_buffer(8192)
            options = (INT * 2)(5, 6)  # CU_JIT_ERROR_LOG_BUFFER, _SIZE_BYTES
            values = (PTR * 2)(c.addressof(error_log), len(error_log))
            try:
                call('cuModuleLoadDataEx', c.byref(module), ptx, 2, options, values)
            except CudaError:
                emit(kind='jit_error', log=error_log.value.decode(errors='replace'))
                raise
            function = PTR()
            call('cuModuleGetFunction', c.byref(function), module, b'load_ragged_only')

            # Only [base, base + backing_bytes) is accessible after base.
            # The preceding mapped bytes do not satisfy the LARGE-mode requirement.
            backing_bytes = (128 if case == 'backed' else 64) * 1024
            prop = AllocationProp(type=1, location=Location(1, device.value))
            granularity = SIZE()
            call('cuMemGetAllocationGranularity', c.byref(granularity), c.byref(prop), 0)
            mapped_bytes = (backing_bytes + granularity.value - 1) // granularity.value * granularity.value
            address, handle = U64(), U64()
            call('cuMemAddressReserve', c.byref(address), mapped_bytes + granularity.value, 0, 0, 0)
            call('cuMemCreate', c.byref(handle), mapped_bytes, c.byref(prop), 0)
            call('cuMemMap', address, mapped_bytes, 0, handle, 0)
            access = AccessDesc(Location(1, device.value), 3)
            call('cuMemSetAccess', address, mapped_bytes, c.byref(access), 1)
            base = address.value + mapped_bytes - backing_bytes
            source = bytes((row * 13 + col * 7) % 251 + 1 for row in range(32) for col in range(2048))
            host_source = c.create_string_buffer(source)
            call('cuMemcpyHtoD_v2', base, host_source, len(source))

            host_map = c.create_string_buffer(128 + 63)
            aligned_map = (c.addressof(host_map) + 63) & ~63
            shape = (U64 * 5)(2048, 1 << 30, 1, 0x7fff0000, 0x7fff0000)
            strides = (U64 * 4)(2048, 0, 2048, (1 << 34) - 2048)
            box, element_strides = (U32 * 5)(128, 128, 1, 1, 1), (U32 * 5)(1, 1, 1, 1, 1)
            call('cuTensorMapEncodeTiled', aligned_map, 0, 5, base, shape, strides,
                 box, element_strides, 0, 0, 2, 0)
            words = (U64 * 16).from_address(aligned_map)
            original_class = 'LARGE' if words[1] & (1 << 21) else 'SMALL'
            # NVIDIA's documented Blackwell descriptor correction:
            # https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-595-91-07/index.html#known-issues
            if case == 'small':
                words[1] &= ~(1 << 21)
            elif case == 'large':
                words[1] |= 1 << 21
            effective_class = 'LARGE' if words[1] & (1 << 21) else 'SMALL'
            emit(kind='descriptor', original_class=original_class, effective_class=effective_class,
                 class_read_method='Blackwell descriptor bit 85', base=hex(base),
                 guard=hex(address.value + mapped_bytes), accessible_bytes_from_base=backing_bytes,
                 physical_shape=[1, 32, 2048], physical_strides=[0, 2048, 1],
                 cuda_dimensions=list(shape), cuda_byte_strides=list(strides),
                 cuda_box=list(box), slice_offset=0, live_rows=1)

            descriptor, output = U64(), U64()
            call('cuMemAlloc_v2', c.byref(descriptor), 128)
            call('cuMemAlloc_v2', c.byref(output), 16384)
            call('cuMemcpyHtoD_v2', descriptor, aligned_map, 128)
            live_rows = U32(1)
            args = (PTR * 3)(c.addressof(descriptor), c.addressof(output), c.addressof(live_rows))
            stage = 'load'
            call('cuLaunchKernel', function, 1, 1, 1, 32, 1, 1, 0, None, args, None)
            call('cuCtxSynchronize')
            stage = 'check'
            result = c.create_string_buffer(16384)
            call('cuMemcpyDtoH_v2', result, output, 16384)
            if result.raw != source[:128] + bytes(127 * 128):
                raise RuntimeError('Output differs from one live row followed by zero-filled rows.')
            emit(kind='result', case=case, status='PASS', effective_class=effective_class)
            return 0
        finally:
            # Each case owns its process. In particular, never reuse a poisoned
            # CUDA context for the next case; process exit also releases VMM resources.
            driver.cuCtxDestroy_v2(context)
    except Exception as error:
        illegal = isinstance(error, CudaError) and error.code == 700 and stage == 'load'
        emit(kind='result', case=case, status='ILLEGAL_ADDRESS' if illegal else 'ERROR',
             effective_class=effective_class, stage=stage, error=str(error))
        return 1


def run_suite(ordinal):
    results = {}
    for case in ('original', 'small', 'large', 'backed'):
        print(f'\n--- {case} ---', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--case', case, '--device', str(ordinal)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            results[case] = {'status': 'ERROR', 'error': 'case timed out after 120 seconds'}
            emit(kind='result', case=case, **results[case])
            continue
        print(completed.stdout, end='', flush=True)
        if completed.stderr:
            print(completed.stderr, end='', file=sys.stderr, flush=True)
        for line in completed.stdout.splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get('kind') == 'result':
                results[case] = record
        if case not in results:
            results[case] = {'status': 'ERROR', 'error': f'child exited {completed.returncode} without a result'}
    controls_pass = all(results[case]['status'] == 'PASS' for case in ('small', 'backed'))
    original_faults = results['original']['status'] == 'ILLEGAL_ADDRESS'
    forced_faults = results['large']['status'] == 'ILLEGAL_ADDRESS'
    if controls_pass and original_faults and forced_faults:
        print('\nCONFIRMED: original/LARGE + 64 KiB faults; SMALL and 128 KiB controls pass.', flush=True)
        return 0
    if controls_pass and forced_faults and results['original']['status'] == 'PASS':
        print('\nFORCED-ONLY: LARGE + 64 KiB faults, but the original driver descriptor passes. '
              'Check its reported class; this does not reproduce the original driver-selection bug.', flush=True)
        return 2
    if controls_pass and all(results[case]['status'] == 'PASS' for case in ('original', 'large')):
        print('\nNOT REPRODUCED on this environment. Share the environment and descriptor records above.', flush=True)
        return 2
    print('\nINCONCLUSIVE: a setup/control failed or outcomes differ. Share the complete output.', flush=True)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all', 'original', 'small', 'large', 'backed'], default='all')
    parser.add_argument('--device', type=int, default=0, help='CUDA-visible device ordinal (default: 0)')
    args = parser.parse_args()
    return run_suite(args.device) if args.case == 'all' else run_case(args.case, args.device)


if __name__ == '__main__':
    sys.exit(main())

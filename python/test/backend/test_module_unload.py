import triton.compiler.compiler as compiler
from triton import knobs
import triton
import tempfile
from triton._internal_testing import is_hip


def test_module_load_unload():

    def get_current_target():
        driver = triton.runtime.driver.active
        device = driver.get_current_device()
        backend = 'hip' if is_hip() else 'cuda'
        if is_hip():
            device_properties = driver.utils.get_device_properties(device)
            warp_size = device_properties['warpSize']
            arch = knobs.runtime.override_arch or device_properties['arch']
            capability = arch.split(':')[0]
        else:
            warp_size = 32
            capability = driver.get_device_capability(device)
            capability = capability[0] * 10 + capability[1]

        return compiler.GPUTarget(backend, capability, warp_size)

    add_kernel_ttir = '''
#loc1 = loc("x_ptr")
#loc2 = loc("y_ptr")
#loc3 = loc("output_ptr")
#loc4 = loc("n_elements")
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("x_ptr"), %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("y_ptr"), %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("output_ptr"), %n_elements: i32 {tt.divisibility = 16 : i32} loc("n_elements")) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc)
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc)
    %5 = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc)
    %7 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc)
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc)
    %10 = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc)
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc)
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc)
    %14 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
    '''

    with tempfile.NamedTemporaryFile(suffix='.ttir') as tmp_file:
        with open(tmp_file.name, "w") as fd_in:
            fd_in.write(add_kernel_ttir)

        print(f"tmp_file = {tmp_file.name}")
        target = get_current_target()
        # compiled_kernel = compiler.compile(str(Path(__file__).parent / "add_kernel.ttir"), target, None, None)
        compiled_kernel = compiler.compile(tmp_file.name, target, None, None)
        compiled_kernel._init_handles()
        print(f"module = {compiled_kernel.module}")
        assert compiled_kernel.module is not None
        compiled_kernel.__del__()
        assert compiled_kernel.module is None

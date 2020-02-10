===========================
Writing a Custom Operation
===========================

--------------
Compute Kernel
--------------

Let us start with something simple, and see how Triton can be used to create a custom vector addition for PyTorch. The Triton compute kernel for this operation is the following:

.. code-block:: C

    // Triton
    // launch on a grid of (N  + TILE - 1) / TILE programs
    __global__ void add(float* z, float* x, float* y, int N){
        // program id
        int pid = get_program_id(0);
        // create arrays of pointers
        int offset[TILE] = pid * TILE + 0 ... TILE;
        float* pz[TILE] = z + offset;
        float* px[TILE] = x + offset;
        float* py[TILE] = y + offset;
        // bounds checking
        bool check[TILE] = offset < N;
        // write-back
        *?(check)pz = *?(check)*px + *?(check)py;
    }

As you can see, arrays are first-class citizen in Triton. This has a number of important advantages that will be highlighted in the next tutorial. For now, let's keep it simple and see how to execute the above operation in PyTorch.

---------------
PyTorch Wrapper
---------------

As you will see, a wrapper for the above Triton function can be created in just a few lines of pure python code.

.. code-block:: python

    import torch
    import triton

    class _add(triton.function):
        # source-code for Triton compute kernel
        src = """
    __global__ void add(float* z, float* x, float* y, int N){
        // program id
        int pid = get_program_id(0);
        // create arrays of pointers
        int offset[TILE] = pid * TILE + 0 ... TILE;
        float* pz[TILE] = z + offset;
        float* px[TILE] = x + offset;
        float* py[TILE] = y + offset;
        // bounds checking
        bool check[TILE] = offset < N;
        // write-back
        *?(check)pz = *?(check)*px + *?(check)py;
    }
        """
        # create callable kernel for the source-code
        kernel = triton.kernel(src)

        # Forward pass
        @staticmethod
        def forward(ctx, x, y):
            # type checking
            assert x.dtype == torch.float32
            # allocate output
            z = torch.empty_like(x).cuda()
            # create launch grid
            # this is a function of the launch parameters
            # triton.cdiv indicates ceil division
            N = x.numel()
            grid = lambda opt: (triton.cdiv(N, opt.d('TILE')), )
            # launch kernel
            # options: 4 warps and a -DTILE=1024
            _add.kernel(z, x, y, N, 
                        grid = grid, 
                        num_warps = 4,
                        defines = {'TILE': 1024})
            # return output
            return z

    # get callable from Triton function
    add = _add.apply

    # test
    torch.manual_seed(0)
    x = torch.rand(98432).cuda()
    y = torch.rand(98432).cuda()
    za = x + y
    zb = add(x, y)
    diff = (za - zb).abs().max()
    print(diff)

Executing the above code will:

- Generate a .cpp file containing PyTorch bindings for the Triton function
- Compile this .cpp file using distutils
- Cache the resulting custom op
- Call the resulting custom op

In other words, the first program run will generate and cache a bunch of files in $HOME/.triton/cache, but subsequent runs should be just as fast as using a handwritten custom operation.
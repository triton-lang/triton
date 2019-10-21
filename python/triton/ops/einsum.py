import triton

class _einsum(triton.function):

    src = """
    void einsum(TYPE * A, TYPE * B, TYPE * C,
                int dim_M, int dim_N, int dim_K, 
                int std_A0, int std_B0, int std_C0, 
                int std_A1, int std_B1, int std_C1) {
        int pid0 = get_program_id(0);
        int pid1 = get_program_id(1);
    }
    """

    kernel = triton.kernel(src, ['C'])
  
    @staticmethod
    def _append_dim(dim_data, dim_type, idx, label, dim, stride):
        if dim_type in dim_data:
            data = dim_data[dim_type]
            if idx != data["idx"] + 1:
                raise ValueError("aggregate inner, outer and batch dims must be adjacent to each other.")
            data["dim"] *= dim
            data["lab"]  = label + data["lab"]
        else:
            dim_data[dim_type] = dict(idx=idx, lab=label, dim=dim, std=stride)
        return dim_type

    @staticmethod
    def _parse_abc(labels_a, labels_b, labels_c, shape_a, is_a=False):

        if len(labels_a) != len(shape_a):
            raise ValueError(f"einsum notation dims do not match shape: {labels_a} {shape_a}")

        trans  = False
        stride = 1
        std1   = None
        data   = dict()
        for idx, (lab, dim) in enumerate(reversed(list(zip(labels_a, shape_a)))):
            #print(idx, lab, dim)
            if dim is None:
                raise ValueError("einsum doens't currently work on shapes with placeholder dims.")
            if idx == 0 and dim % 8 != 0:
                raise ValueError("contiguous dim must be multiple of 8")

            if lab in labels_c:
                # batch dim
                if lab in labels_b:
                    _einsum._append_dim(data, "B", idx, lab, dim, stride)
                    if idx == 0:
                        raise ValueError(f"batch dim can not be contiguous dim: {lab} {labels_a} {shape_a}")
                # outer dim
                else:
                    std1 = _einsum._append_dim(data, "O", idx, lab, dim, stride)
                    if idx == 0:
                        trans = is_a
            # inner dim
            elif lab in labels_b:
                std1 = _einsum._append_dim(data, "I", idx, lab, dim, stride)
                if idx == 0:
                    trans = not is_a
            else:
                raise ValueError(f"einsum def for output: {lab} ({labels_a}), not present in either other def")

            stride *= dim

        if "B" not in data:
            data["B"] = dict(dim=1, std=1)

        # batch, outer, inner, std0, std1, trans
        return data["B"]["dim"], data["O"]["dim"], data["I"]["dim"], data["B"]["std"], data[std1]["std"], trans

    @staticmethod
    def _parse_einsum(labels_a, labels_b, labels_c, shape_a, shape_b):

        dims_a  = dict(zip(labels_a, shape_a))
        dims_b  = dict(zip(labels_b, shape_b))
        shape_c = list()
        for lab in labels_c:
            if lab in dims_a:
                shape_c.append(dims_a[lab])
            elif lab in dims_b:
                shape_c.append(dims_b[lab])
            else:
                raise ValueError(f"einsum def for output: {lab} ({labels_c}), not present in either input def ({labels_a}, {labels_b})")

        BA, M, KA, std_a0, std_a1, ta = _einsum._parse_abc(labels_a, labels_b, labels_c, shape_a, True)
        BB, N, KB, std_b0, std_b1, tb = _einsum._parse_abc(labels_b, labels_a, labels_c, shape_b, False)
        BC, _,  _, std_c0, std_c1,  _ = _einsum._parse_abc(labels_c, labels_b, labels_a, shape_c)

        if not (BA == BB == BC):
            raise ValueError("mismatched batch dims")
        if KA != KB:
            raise ValueError("mismatched reduction dims")

        return shape_c, (BA, M, N, KA), (std_a0, std_b0, std_c0), (std_a1, std_b1, std_c1), ta, tb

    @staticmethod
    def call(a, b, trans_a, trans_b, shape_c, bmnk,
             std0, std1, einsum_a, einsum_b, einsum_c):
        dtype = a.dtype
        c = triton.empty(shape_c, dtype)
        grid = lambda opt: (1, 1, 1)
        return _einsum.kernel(a, b, c, 
                              bmnk[1], bmnk[2], bmnk[3], 
                              std0[0], std0[1], std0[2], 
                              std1[0], std1[1], std1[2], 
                              grid,
                              TYPE=['float'])
    

    @staticmethod
    def forward(ctx, subscripts, a, b):
        if type(subscripts) is str:
            einsum_a, einsum_bc = subscripts.split(",")
            einsum_b, einsum_c  = einsum_bc.split("->")
        else:
            einsum_a, einsum_b, einsum_c = subscripts

        shape_c, bmnk, std0, std1, ta, tb = _einsum._parse_einsum(
                                                einsum_a, einsum_b, einsum_c,
                                                a.shape, b.shape
                                                )
        return _einsum.call(a, b, ta, tb, shape_c, bmnk, std0, std1, einsum_a, einsum_b, einsum_c)
        
einsum = _einsum.apply
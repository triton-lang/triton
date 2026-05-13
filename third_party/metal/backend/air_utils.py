import re


def rewrite_ptrs(ir: str) -> str:
    # nuw flag not supported
    # prev: getelementptr inbounds nuw (...)
    # new:  getelementptr inbounds (...)
    ir = re.sub(r"getelementptr inbounds nuw \(", "getelementptr inbounds (", ir)

    # prev: getelementptr inbounds (i8, ptr addrspace(3) @global_smem, i64 16384)
    # new: getelementptr inbounds (i8, i8 addrspace(3)* @global_smem, i64 16384)
    ir = re.sub(
        r"(getelementptr(?:\s+inbounds)?)\s+\((<\d+\s+x\s+\w+>|\w+),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)} ({m.group(2)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    # prev: getelementptr float, ptr addrspace(1) %0, i64 %9
    # new: getelementptr float, float addrspace(1)* %0, i64, %9
    ir = re.sub(
        r"(getelementptr(?:\s+inbounds)?\s+(<\d+\s+x\s+\w+>|\w+)),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    # prev: load float, ptr addrspace(1) %10
    # new: load float, float addrspace(1)* %10
    ir = re.sub(
        r"(load\s+(<\d+\s+x\s+\w+>|\w+)),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    # prev: store float %10, ptr addrspace(1) %15
    # new: store float %10, float addrspace(1)* %15
    ir = re.sub(
        r"(store\s+(<\d+\s+x\s+\w+>|\w+)\s+[^,\n]+),\s+ptr\s+addrspace\((\d+)\)",
        lambda m: f"{m.group(1)}, {m.group(2)} addrspace({m.group(3)})*",
        ir,
    )

    return ir


def generate_getelementptr_type_dict(ir: str) -> dict:
    getelementptr_type_dict: dict[
        str, tuple[str, str]
    ] = {}  # map ptr name (result of getelementptr) to ptr type and addr space
    for gep_match in re.finditer(
        r"(\s*)(%\w+)\s*=\s*getelementptr\b[^,\n]*,\s+(<\d+\s+x\s+\w+>|\w+)\s+addrspace\((\d+)\)\*",
        ir,
    ):
        # group(1): leading spaces
        # group(2): ptr var name, result of getelementptr
        # group(3): ptr type
        # group(4): addr space
        ptr_var_name = gep_match.group(2)
        ptr_type = gep_match.group(3)
        addrspace = gep_match.group(4)
        getelementptr_type_dict[ptr_var_name] = (ptr_type, addrspace)

    return getelementptr_type_dict


def insert_bitcast_for_vecs(ir: str, getelementptr_type_dict: dict, struct_type_dict: dict) -> str:
    # insert bitcast when loading vector from scalar ptr that is result of getelementptr
    # e.g.
    # %54 = getelementptr float, float addrspace(1)* %52, i64 %53
    # %62 = load <1 x float>, <1 x float> addrspace(1)* %54, align 4

    # get type of @global_smem
    global_smem_definition_match = re.search(
        r"@global_smem\s*=\s*\w+\s+addrspace\(\d+\)\s+global\s+(\[\d+\s+x\s+\w+\])", ir
    )
    global_smem_type = None
    if global_smem_definition_match:
        global_smem_type = global_smem_definition_match.group(1)
    else:
        raise RuntimeError("@global_smem definition not found")

    cast_idx = 0
    new_lines = []
    for line in ir.split("\n"):
        gep_match_global_smem = re.match(
            r"(\s*)(%\w+)\s*=\s*getelementptr\b[^,\n]*,\s+(<\d+\s+x\s+\w+>|\w+)\s+addrspace\((\d+)\)\*\s+([@%]\w+)(.*)",
            line,
        )
        if gep_match_global_smem:
            leading_spaces = gep_match_global_smem.group(1)
            ptr_var_name = gep_match_global_smem.group(2)
            ptr_type = gep_match_global_smem.group(3)
            addrspace = gep_match_global_smem.group(4)
            base_ptr = gep_match_global_smem.group(5)
            remaining = gep_match_global_smem.group(6)
            if int(addrspace) == 3 and base_ptr == "@global_smem":
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_type} addrspace({addrspace})* {base_ptr} to {ptr_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{ptr_var_name} = getelementptr inbounds {ptr_type}, {ptr_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue

        # search for load from scalar pointer that is the result of getelementptr or extractvalue
        load_match = re.match(
            r"(\s*)(%\w+)\s*=\s*load\s+(<\d+\s+x\s+(\w+)>),\s+<[^>]+>\s+addrspace\((\d+)\)\*\s+(%\w+)(.*)",
            line,
        )
        if load_match:
            leading_spaces = load_match.group(1)
            result_var = load_match.group(2)
            vec_type = load_match.group(3)
            scalar_elem = load_match.group(4)
            addrspace = load_match.group(5)
            ptr = load_match.group(6)
            remaining = load_match.group(7)
            if getelementptr_type_dict.get(ptr) == (scalar_elem, addrspace):
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_elem} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{result_var} = load {vec_type}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif struct_type_dict.get(ptr) == f"{scalar_elem} addrspace({addrspace})*":
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_elem} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}{result_var} = load {vec_type}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif int(addrspace) == 3 and getelementptr_type_dict.get(ptr, (None, None))[0] != vec_type:
                # if loading from global smem (addrspace 3), then cast global smem ptr
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                global_smem_ptr_type, global_smem_addrspace = getelementptr_type_dict[ptr]
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_ptr_type} addrspace({global_smem_addrspace})* {ptr} to {vec_type} addrspace({global_smem_addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}load {vec_type}, {vec_type} addrspace({global_smem_addrspace})* {cast_result}{remaining}"
                )
                continue

        # search for store to scalar pointer that is the result of getelementptr
        store_match = re.match(
            r"(\s*)store\s+(<\d+\s+x\s+(\w+)>)\s+([^,]+),\s+<[^>]+>\s+addrspace\((\d+)\)\*\s+(%\w+)(.*)",
            line,
        )
        if store_match:
            leading_spaces = store_match.group(1)
            vec_type = store_match.group(2)
            scalar_type = store_match.group(3)
            val = store_match.group(4)
            addrspace = store_match.group(5)
            ptr = store_match.group(6)
            remaining = store_match.group(7)
            if getelementptr_type_dict.get(ptr) == (scalar_type, addrspace):
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {scalar_type} addrspace({addrspace})* {ptr} to {vec_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}store {vec_type} {val}, {vec_type} addrspace({addrspace})* {cast_result}{remaining}"
                )
                continue
            elif int(addrspace) == 3 and getelementptr_type_dict.get(ptr, (None, None))[0] != vec_type:
                # if storing into global smem (addrspace 3), then cast global smem ptr
                cast_result = f"%vec_cast_{cast_idx}"
                cast_idx += 1
                global_smem_ptr_type, global_smem_addrspace = getelementptr_type_dict[ptr]
                new_lines.append(
                    f"{leading_spaces}{cast_result} = bitcast {global_smem_ptr_type} addrspace({global_smem_addrspace})* {ptr} to {vec_type} addrspace({global_smem_addrspace})*"
                )
                new_lines.append(
                    f"{leading_spaces}store {vec_type} {val}, {vec_type} addrspace({global_smem_addrspace})* {cast_result}{remaining}"
                )
                continue

        new_lines.append(line)
    ir = "\n".join(new_lines)
    return ir


def insert_bitcast_for_global_smem_load_store(ir: str) -> str:
    # e.g. store i32 %1425, i32 addrspace(3)* @global_smem, align 4
    # e.g. %1427 = load i32, i32 addrspace(3)* @global_smem, align 4
    global_smem_match = re.search(r"@global_smem\s*=\s*\w+\s+addrspace\(\d+\)\s+global\s+(\[\d+\s+x\s+\w+\])", ir)
    if not global_smem_match:
        return ir

    global_smem_type = global_smem_match.group(1)
    cast_idx = 0
    new_lines = []
    for line in ir.split("\n"):
        # store type %val, type addrspace(3)* @global_smem ...
        store_match = re.match(
            r"(?P<indent>\s*)store\s+(?P<type><\d+\s+x\s+\w+>|\w+)\s+(?P<val>[^,]+),\s+(?P<type2><[^>]+>|\w+)\s+addrspace\(3\)\*\s+@global_smem(?P<rest>.*)",
            line,
        )
        if store_match:
            elem_type = store_match.group("type")
            cast_var = f"%smem_cast_{cast_idx}"
            cast_idx += 1
            new_lines.append(
                f"{store_match.group('indent')}{cast_var} = bitcast {global_smem_type} addrspace(3)* @global_smem to {elem_type} addrspace(3)*"
            )
            new_lines.append(
                f"{store_match.group('indent')}store {elem_type} {store_match.group('val').strip()}, {elem_type} addrspace(3)* {cast_var}{store_match.group('rest')}"
            )
            continue

        # %res = load type, type addrspace(3)* @global_smem ...
        load_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*load\s+(?P<type><\d+\s+x\s+\w+>|\w+),\s+(?P<type2><[^>]+>|\w+)\s+addrspace\(3\)\*\s+@global_smem(?P<rest>.*)",
            line,
        )
        if load_match:
            elem_type = load_match.group("type")
            cast_var = f"%smem_cast_{cast_idx}"
            cast_idx += 1
            new_lines.append(
                f"{load_match.group('indent')}{cast_var} = bitcast {global_smem_type} addrspace(3)* @global_smem to {elem_type} addrspace(3)*"
            )
            new_lines.append(
                f"{load_match.group('indent')}{load_match.group('result')} = load {elem_type}, {elem_type} addrspace(3)* {cast_var}{load_match.group('rest')}"
            )
            continue

        new_lines.append(line)
    return "\n".join(new_lines)


def insert_bitcast_for_global_smem_gep(ir: str) -> str:
    """
    Before:
        %r = getelementptr inbounds i8, i8 addrspace(3)* getelementptr inbounds (i8, i8 addrspace(3)* @global_smem, i64 16384), i32 %idx
    After:
        %smem_gep_cast_N = bitcast [K x i8] addrspace(3)* @global_smem to i8 addrspace(3)*
        %smem_gep_base_N = getelementptr inbounds i8, i8 addrspace(3)* %smem_gep_cast_N, i64 16384
        %r               = getelementptr inbounds i8, i8 addrspace(3)* %smem_gep_base_N, i32 %idx
    """
    global_smem_match = re.search(r"@global_smem\s*=\s*\w+\s+addrspace\(\d+\)\s+global\s+(\[\d+\s+x\s+\w+\])", ir)
    if not global_smem_match:
        return ir
    global_smem_type = global_smem_match.group(1)

    cast_idx = 0
    new_lines = []
    for line in ir.split("\n"):
        # inbounds is optional on outer GEP instr
        # inner_elem may differ from outer elem_type (e.g. i8 inner, float outer)
        m = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*getelementptr(?P<inbounds>\s+inbounds)? (?P<elem_type>\w+),\s*"
            r"(?P<ptr_type>\w+)\s+addrspace\((?P<addrspace>\d+)\)\*\s+"
            r"getelementptr(?:\s+inbounds)?\s+\((?P<inner_elem>\w+),\s+\w+\s+addrspace\(\d+\)\*\s+@global_smem,\s+i64\s+(?P<offset>\d+)\)"
            r"(?P<remaining>.*)",
            line,
        )
        if m:
            indent = m.group("indent")
            result = m.group("result")
            inbounds = m.group("inbounds") or ""
            elem_type = m.group("elem_type")
            addrspace = m.group("addrspace")
            inner_elem = m.group("inner_elem")
            offset = m.group("offset")
            remaining = m.group("remaining")

            cast_var = f"%smem_gep_cast_{cast_idx}"
            base_var = f"%smem_gep_base_{cast_idx}"
            cast_idx += 1

            # bitcast global array to inner element ptr (always i8 for byte GEPs)
            new_lines.append(
                f"{indent}{cast_var} = bitcast {global_smem_type} addrspace({addrspace})* @global_smem to {inner_elem} addrspace({addrspace})*"
            )
            new_lines.append(
                f"{indent}{base_var} = getelementptr inbounds {inner_elem}, {inner_elem} addrspace({addrspace})* {cast_var}, i64 {offset}"
            )

            # if outer GEP uses different elem type, add bitcast
            outer_base_var = base_var
            if inner_elem != elem_type:
                typed_var = f"%smem_gep_typed_{cast_idx - 1}"
                new_lines.append(
                    f"{indent}{typed_var} = bitcast {inner_elem} addrspace({addrspace})* {base_var} to {elem_type} addrspace({addrspace})*"
                )
                outer_base_var = typed_var

            new_lines.append(
                f"{indent}{result} = getelementptr{inbounds} {elem_type}, {elem_type} addrspace({addrspace})* {outer_base_var}{remaining}"
            )
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


def insert_bitcast_for_global_smem_call(ir: str) -> str:
    """Materialize constexpr GEP off @global_smem in call instr

    Constexpr GEP can't reference SSA values, so need to expand calls that passes SSA val as arg

    Before:
        %r = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p3f16(ptr addrspace(3) getelementptr inbounds (i8, i8 addrspace(3)* @global_smem, i64 16384), ...)
    After:
        %smem_call_gep_cast_N  = bitcast [K x i8] addrspace(3)* @global_smem to i8 addrspace(3)*
        %smem_call_gep_base_N  = getelementptr inbounds i8, i8 addrspace(3)* %smem_call_gep_cast_N, i64 16384
        %smem_call_gep_typed_N = bitcast i8 addrspace(3)* %smem_call_gep_base_N to half addrspace(3)*
        %r = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p3f16(half addrspace(3)* %smem_call_gep_typed_N, ...)
    """
    global_smem_match = re.search(r"@global_smem\s*=\s*\w+\s+addrspace\(\d+\)\s+global\s+(\[\d+\s+x\s+\w+\])", ir)
    if not global_smem_match:
        return ir
    global_smem_type = global_smem_match.group(1)

    constexpr_gep = re.compile(
        r"(ptr\s+addrspace\((\d+)\))\s+getelementptr inbounds\s+\((\w+),\s+\w+\s+addrspace\(\d+\)\*\s+@global_smem,\s+i64\s+(\d+)\)"
    )
    # same pattern but if outer ptr already typed
    # e.g. after rewrite_air_simdgroup_async_copy_ptrs converts ptr addrspace(3) to i8 addrspace(3)*
    typed_constexpr_gep = re.compile(
        r"(\w+\s+addrspace\((\d+)\)\*)\s+getelementptr inbounds\s+\((\w+),\s+\w+\s+addrspace\(\d+\)\*\s+@global_smem,\s+i64\s+(\d+)\)"
    )
    simdgroup_func = re.compile(r"@(air\.simdgroup_matrix_8x8_(?:load|store)\.[^(]+)\(")

    # also matches simdgroup calls that pass @global_smem directly (no GEP offset)
    direct_global_smem = re.compile(r"ptr\s+addrspace\((\d+)\)\s+@global_smem")

    cast_idx = 0
    new_lines = []
    for line in ir.split("\n"):
        if "call " not in line or "@global_smem" not in line:
            new_lines.append(line)
            continue

        indent = re.match(r"(\s*)", line).group(1)
        sg_m = simdgroup_func.search(line)

        # get typed ptr from simdgroup function name suffix
        def simdgroup_typed_ptr(func_name):
            ptr_suffix_m = re.search(r"\.p(\d+)(\w+)$", func_name)
            if ptr_suffix_m:
                llvm_type = _AIR_ELEM_TYPE_MAP.get(ptr_suffix_m.group(2))
                if llvm_type:
                    return f"{llvm_type} addrspace({ptr_suffix_m.group(1)})*"
            return None

        # case 1: constexpr GEP off @global_smem as call arg (opaque or typed outer ptr)
        if "getelementptr inbounds" in line:
            m = constexpr_gep.search(line)
            matched_gep_pat = constexpr_gep
            if not m:
                m = typed_constexpr_gep.search(line)
                matched_gep_pat = typed_constexpr_gep
            if m:
                addrspace = m.group(2)
                elem_type = m.group(3)  # i8
                offset = m.group(4)

                cast_var = f"%smem_call_gep_cast_{cast_idx}"
                base_var = f"%smem_call_gep_base_{cast_idx}"
                cast_idx += 1

                new_lines.append(
                    f"{indent}{cast_var} = bitcast {global_smem_type} addrspace({addrspace})* @global_smem to {elem_type} addrspace({addrspace})*"
                )
                new_lines.append(
                    f"{indent}{base_var} = getelementptr inbounds {elem_type}, {elem_type} addrspace({addrspace})* {cast_var}, i64 {offset}"
                )

                arg_var = base_var
                arg_ptr_ty = f"{elem_type} addrspace({addrspace})*"
                if sg_m:
                    typed_ptr = simdgroup_typed_ptr(sg_m.group(1))
                    if typed_ptr:
                        typed_var = f"%smem_call_gep_typed_{cast_idx - 1}"
                        new_lines.append(f"{indent}{typed_var} = bitcast {arg_ptr_ty} {base_var} to {typed_ptr}")
                        arg_var = typed_var
                        arg_ptr_ty = typed_ptr

                new_lines.append(matched_gep_pat.sub(f"{arg_ptr_ty} {arg_var}", line, count=1))
                continue

        # case 2: @global_smem passed directly as ptr addrspace(N) arg in simdgroup call
        if sg_m:
            dm = direct_global_smem.search(line)
            if dm:
                addrspace = dm.group(1)
                typed_ptr = simdgroup_typed_ptr(sg_m.group(1))
                if typed_ptr:
                    cast_var = f"%smem_call_direct_cast_{cast_idx}"
                    cast_idx += 1
                    new_lines.append(
                        f"{indent}{cast_var} = bitcast {global_smem_type} addrspace({addrspace})* @global_smem to {typed_ptr}"
                    )
                    new_lines.append(direct_global_smem.sub(f"{typed_ptr} {cast_var}", line, count=1))
                    continue

        # case 3: @global_smem already has a typed ptr
        # e.g. after rewrite_air_simdgroup_async_copy_ptrs converts ptr addrspace(3) @global_smem to i8 addrspace(3)* @global_smem
        typed_global_smem = re.search(r"(\w+)\s+addrspace\((\d+)\)\*\s+@global_smem", line)
        if typed_global_smem:
            elem_type = typed_global_smem.group(1)
            addrspace = typed_global_smem.group(2)
            typed_ptr = f"{elem_type} addrspace({addrspace})*"
            cast_var = f"%smem_call_direct_cast_{cast_idx}"
            cast_idx += 1
            new_lines.append(
                f"{indent}{cast_var} = bitcast {global_smem_type} addrspace({addrspace})* @global_smem to {typed_ptr}"
            )
            new_lines.append(
                re.sub(
                    rf"{re.escape(elem_type)}\s+addrspace\({re.escape(addrspace)}\)\*\s+@global_smem",
                    f"{typed_ptr} {cast_var}",
                    line,
                    count=1,
                )
            )
            continue

        new_lines.append(line)

    return "\n".join(new_lines)


def get_func_args(s: str) -> list:
    """Get arg list, handling nested parentheses"""
    args, depth, cur = [], 0, []
    for ch in s:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def modify_func_signature(ir: str, ptr_type_dict: dict) -> str:
    func_start_idx = ir.index("define void @")
    open_paren_idx = ir.index("(", func_start_idx) + 1
    # find closing ")"
    depth, i = 1, open_paren_idx
    while i < len(ir) and depth > 0:
        if ir[i] == "(":
            depth += 1
        elif ir[i] == ")":
            depth -= 1
        i += 1
    paren_close = i - 1

    func_args_str = ir[open_paren_idx:paren_close]
    args = get_func_args(func_args_str)
    new_args = []
    for arg in args:
        m = re.match(r"(ptr addrspace\((\d+)\))(.*?)(%\w+)$", arg.strip())
        if m:
            addrspace = m.group(2)
            attrs = m.group(3)
            var_name = m.group(4)
            typed = ptr_type_dict.get(var_name, f"i8 addrspace({addrspace})*")
            new_args.append(f"{typed}{attrs}{var_name}")
        else:
            new_args.append(arg)
    ir = ir[:open_paren_idx] + ", ".join(new_args) + ir[paren_close:]
    return ir


_AIR_ELEM_TYPE_MAP = {
    "f16": "half",
    "f32": "float",
    "f64": "double",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
}


def rewrite_air_simdgroup_decl_ptrs(ir: str) -> str:
    """Rewrite opaque ptr to typed ptr in air.simdgroup_matrix_8x8_* declarations

    Elem type and addrspace in function name suffix
    e.g. @air.simdgroup_matrix_8x8_load.v64f16.p3f16 means ptr type is half addrspace(3)*
    """
    # Map func name -> typed ptr
    func_ptr_type: dict[str, str] = {}
    for m in re.finditer(
        r"declare\s+[^@]+@(air\.simdgroup_matrix_8x8_(?:load|store)\.[^(]+)\(",
        ir,
    ):
        func_name = m.group(1).strip()
        if func_name in func_ptr_type:
            continue
        ptr_m = re.search(r"\.p(\d+)(\w+)$", func_name)
        if not ptr_m:
            continue
        addrspace = ptr_m.group(1)
        elem_suffix = ptr_m.group(2)
        llvm_type = _AIR_ELEM_TYPE_MAP.get(elem_suffix)
        if not llvm_type:
            continue
        func_ptr_type[func_name] = f"{llvm_type} addrspace({addrspace})*"

    if not func_ptr_type:
        return ir

    for func_name, typed_ptr in func_ptr_type.items():
        opaque = f"ptr addrspace({typed_ptr.split('addrspace(')[1].split(')')[0]})"
        escaped = re.escape(func_name)

        # rewrite declaration, replace opaque ptr wherever it appears in arg list
        # load has ptr first, store has ptr as second arg
        ir = re.sub(
            rf"(declare\s+[^@]+@{escaped}\([^)]*?){re.escape(opaque)}",
            lambda mo, tp=typed_ptr: mo.group(1) + tp,
            ir,
        )

    return ir


def rewrite_metadata(ir: str) -> str:
    # rewrite metadata func ptr reference
    # e.g. ptr @funcname → void (arg types)* @funcname
    func_match = re.search(r"define void @(\w+)\(", ir)
    if func_match:
        func_name = func_match.group(1)
        func_open_str = "define void @" + func_name + "("
        func_signature_start_idx = ir.index(func_open_str) + len(func_open_str)
        depth, j = 1, func_signature_start_idx
        while j < len(ir) and depth > 0:
            if ir[j] == "(":
                depth += 1
            elif ir[j] == ")":
                depth -= 1
            j += 1
        func_signature_close_idx = j - 1

        func_args = get_func_args(ir[func_signature_start_idx:func_signature_close_idx])
        arg_types = []
        for arg in func_args:
            # extract type: "type addrspace(N)*" or just "type"
            m = re.match(r"(<\d+\s+x\s+\w+>|\w+)(\s+addrspace\(\d+\)\*)?", arg.strip())
            arg_types.append(m.group(0) if m else arg.strip())

        func_type = f"void ({', '.join(arg_types)})*"
        ir = re.sub(
            r"\bptr @" + re.escape(func_name) + r"\b",
            func_type + " @" + func_name,
            ir,
        )

    return ir


# TODO can probably combine insertvalue/phi/extractvalue ptr conversions into one pass over the lines in the ir
def convert_opaque_ptrs_insertvalue(ir: str, getelementptr_type_dict: dict) -> tuple[str, dict]:
    new_lines = []

    # contains types for results of insertvalue
    struct_type_dict = {}
    for line in ir.split("\n"):
        insertvalue_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*insertvalue\s+(?P<agg_type>\{[^}]+\})\s+(?P<struct_vals>\S+),\s*(?P<elem_type>ptr\s+addrspace\((?P<addrspace>\d+)\))\s+(?P<elem_val>%\w+),\s*(?P<idx>\d+)",
            line,
        )
        if insertvalue_match:
            elem_val = insertvalue_match.group("elem_val")
            if elem_val in getelementptr_type_dict:
                elem_type, addrspace = getelementptr_type_dict[elem_val]

                typed_ptr = f"{elem_type} addrspace({addrspace})*"
                new_agg_type = insertvalue_match.group("agg_type").replace(
                    "ptr addrspace(" + addrspace + ")", typed_ptr
                )

                new_line = (
                    f"{insertvalue_match.group('indent')}{insertvalue_match.group('result')} = insertvalue {new_agg_type} "
                    f"{insertvalue_match.group('struct_vals')}, {typed_ptr} {insertvalue_match.group('elem_val')}, {insertvalue_match.group('idx')}"
                )
                new_lines.append(new_line)

                # track struct type in dict
                struct_type_dict[insertvalue_match.group("result")] = new_agg_type
            else:
                raise RuntimeError(
                    f"Elem val {elem_val} was not found in getelementptr_type_dict, perhaps this pointer was created from a different operation (not getelementptr)"
                )
        else:
            new_lines.append(line)

    ir = "\n".join(new_lines)
    return ir, struct_type_dict


def convert_opaque_ptrs_phi(ir: str, struct_type_dict: dict) -> tuple[str, dict]:
    new_lines = []
    for line in ir.split("\n"):
        phi_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*phi\s+(?P<agg_type>\{[^}]+\})\s+(?P<pairs>.*)",
            line,
        )
        if phi_match and "ptr addrspace" in phi_match.group("agg_type"):
            incoming_vals = re.findall(r"\[\s*(%\w+),\s*%\w+\s*\]", phi_match.group("pairs"))
            new_agg_type = next((struct_type_dict[v] for v in incoming_vals if v in struct_type_dict), None)
            if new_agg_type:
                new_line = f"{phi_match.group('indent')}{phi_match.group('result')} = phi {new_agg_type} {phi_match.group('pairs')}"
                new_lines.append(new_line)
                # track so downstream can use this result
                struct_type_dict[phi_match.group("result")] = new_agg_type
            else:
                raise RuntimeError(f"Could not determine struct element type for phi: {line.strip()}")
        else:
            new_lines.append(line)

    ir = "\n".join(new_lines)
    return ir, struct_type_dict


def convert_opaque_ptrs_extractvalue(ir: str, struct_type_dict: dict) -> tuple[str, dict]:
    new_lines = []
    for line in ir.split("\n"):
        extractvalue_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*extractvalue\s+(?P<agg_type>\{[^}]+\})\s+(?P<agg_val>%\w+),\s*(?P<idx>\d+)",
            line,
        )
        if extractvalue_match and "ptr addrspace" in extractvalue_match.group("agg_type"):
            agg_val = extractvalue_match.group("agg_val")
            new_agg_type = struct_type_dict.get(agg_val)
            if new_agg_type:
                new_line = f"{extractvalue_match.group('indent')}{extractvalue_match.group('result')} = extractvalue {new_agg_type} {agg_val}, {extractvalue_match.group('idx')}"
                new_lines.append(new_line)
                fields = [f.strip() for f in new_agg_type.strip("{}").split(",")]
                struct_type_dict[extractvalue_match.group("result")] = fields[int(extractvalue_match.group("idx"))]
            else:
                raise RuntimeError(f"Could not determine struct element type for extractvalue: {line.strip()}")
        else:
            new_lines.append(line)
    return "\n".join(new_lines), struct_type_dict


def convert_opaque_ptrs_cmpxchg(ir: str, getelementptr_type_dict: dict) -> str:
    new_lines = []
    for line in ir.split("\n"):
        m = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*(?P<prefix>cmpxchg(?:\s+weak)?)\s+ptr\s+addrspace\((?P<addrspace>\d+)\)\s+(?P<ptr>%\w+),\s+(?P<val_type>\w+)\s+(?P<rest>.*)",
            line,
        )
        if m:
            ptr = m.group("ptr")
            addrspace = m.group("addrspace")
            if ptr in getelementptr_type_dict:
                elem_type, _ = getelementptr_type_dict[ptr]
            else:
                elem_type = m.group("val_type")
            typed_ptr = f"{elem_type} addrspace({addrspace})*"
            new_lines.append(
                f"{m.group('indent')}{m.group('result')} = {m.group('prefix')} "
                f"{typed_ptr} {ptr}, {m.group('val_type')} {m.group('rest')}"
            )
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def convert_opaque_ptrs_atomicrmw(ir: str, getelementptr_type_dict: dict) -> str:
    new_lines = []
    for line in ir.split("\n"):
        # prev: %res = atomicrmw <op> ptr addrspace(N) %ptr, type %val ...
        # new:  %res = atomicrmw <op> type addrspace(N)* %ptr, type %val ...
        m = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*atomicrmw\s+(?P<op>\w+)\s+ptr\s+addrspace\((?P<addrspace>\d+)\)\s+(?P<ptr>%\w+),\s+(?P<val_type>\w+)\s+(?P<rest>.*)",
            line,
        )
        if m:
            ptr = m.group("ptr")
            addrspace = m.group("addrspace")
            if ptr in getelementptr_type_dict:
                elem_type, _ = getelementptr_type_dict[ptr]
            else:
                elem_type = m.group("val_type")
            typed_ptr = f"{elem_type} addrspace({addrspace})*"
            new_lines.append(
                f"{m.group('indent')}{m.group('result')} = atomicrmw {m.group('op')} "
                f"{typed_ptr} {ptr}, {m.group('val_type')} {m.group('rest')}"
            )
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def convert_opaque_ptrs_ptrtoint(ir: str, getelementptr_type_dict: dict) -> str:
    new_lines = []
    for line in ir.split("\n"):
        ptrtoint_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*ptrtoint\s+ptr\s+addrspace\((?P<addrspace>\d+)\)\s+(?P<ptr_val>%\w+)\s+to\s+(?P<int_type>\w+)",
            line,
        )
        if ptrtoint_match:
            ptr_val = ptrtoint_match.group("ptr_val")
            if ptr_val in getelementptr_type_dict:
                elem_type, addrspace = getelementptr_type_dict[ptr_val]
                typed_ptr = f"{elem_type} addrspace({addrspace})*"
                new_line = f"{ptrtoint_match.group('indent')}{ptrtoint_match.group('result')} = ptrtoint {typed_ptr} {ptr_val} to {ptrtoint_match.group('int_type')}"
                new_lines.append(new_line)
            else:
                raise RuntimeError(
                    f"Ptr val {ptr_val} was not found in getelementptr_type_dict for ptrtoint: {line.strip()}"
                )
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


# this is special case since ptr type is only known after the inttoptr operation
def convert_opaque_ptrs_inttoptr(ir: str) -> str:
    # first find all the inttoptr lines and track the result variables (ptrs)
    # map result var to line
    inttoptr_results = {}  # result_var -> (idx, match)
    lines = ir.split("\n")
    for idx, line in enumerate(lines):
        inttoptr_match = re.match(
            r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*inttoptr\s+(?P<int_type>\w+)\s+(?P<int_val>%\w+)\s+to\s+ptr\s+addrspace\((?P<addrspace>\d+)\)",
            line,
        )
        if inttoptr_match:
            result_var = inttoptr_match.group("result")
            inttoptr_results[result_var] = (idx, inttoptr_match)

    # search for where getelementptr contains these inttoptr results as input
    inttoptr_ptr_types = {}
    for gep_match in re.finditer(
        r"(?P<indent>\s*)(?P<result>%\w+)\s*=\s*getelementptr(?P<inbounds>\s+inbounds)?\s+(?P<elem_type>\w+),\s+(?P<ptr_type>\w+)\s+addrspace\((?P<addrspace>\d+)\)\*\s+(?P<ptr_val>[%@]\w+),\s+(?P<idx_type>\w+)\s+(?P<idx_val>[%@]\w+)",
        ir,
    ):
        ptr_val = gep_match.group("ptr_val")
        if ptr_val in inttoptr_results:
            addrspace = gep_match.group("addrspace")
            inttoptr_ptr_types[ptr_val] = f"{gep_match.group('elem_type')} addrspace({addrspace})*"

    new_lines = lines[:]
    for result_ptr, (idx, m) in inttoptr_results.items():
        assert result_ptr in inttoptr_ptr_types, f"Result ptr {result_ptr} not found in inttoptr_ptr_types"
        ptr_type = inttoptr_ptr_types[result_ptr]
        new_lines[idx] = (
            f"{m.group('indent')}{m.group('result')} = inttoptr {m.group('int_type')} {m.group('int_val')} to {ptr_type}"
        )

    return "\n".join(new_lines)


def convert_opaque_ptrs_to_typed(ir: str) -> str:
    """Convert opaque ptrs to typed ptrs

    Triton LLVM produces opaque ptrs, but metal's JIT can't compile for some reason.
    Metal likely uses older version of LLVM. Need to convert opaque ptrs to typed ptr (e.g. float addrspace(1)*).
    Determine types from load/store/getelementptr.

    Also need to:
    - cast scalar ptrs to vector ptrs when loading vector from scalar ptr.
    - modify function signature
    - modify metadata containing function ptr
    """
    # search for getelementptr/load/store instructions to determine types of the ptrs
    ptr_type_dict: dict = {}
    # e.g. getelementptr/load float, ptr addrspace(1) %1
    for m in re.finditer(
        r"(?:getelementptr(?:\s+inbounds)?|load)\s+(<\d+\s+x\s+\w+>|\w+),\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)", ir
    ):
        # group(3): var name
        # group(1): ptr type
        # group(2): addr space
        ptr_type_dict[m.group(3)] = f"{m.group(1)} addrspace({m.group(2)})*"
    # e.g. store float %val, ptr addrspace(N) %ptr
    for m in re.finditer(r"\bstore\s+(<\d+\s+x\s+\w+>|\w+)\s+[^,\n]+,\s+ptr\s+addrspace\((\d+)\)\s+(%\w+)", ir):
        # group(3): ptr name
        # group(1): type being stored at the ptr
        # group(2): addr space
        ptr_type_dict[m.group(3)] = f"{m.group(1)} addrspace({m.group(2)})*"

    # rewrite opaque ptrs in air.simdgroup_matrix_8x8_* declarations
    ir = rewrite_air_simdgroup_decl_ptrs(ir)

    # rewrite ptrs to include types
    ir = rewrite_ptrs(ir)

    # materialize constant-expression GEPs off @global_smem into real instructions
    ir = insert_bitcast_for_global_smem_gep(ir)
    ir = insert_bitcast_for_global_smem_call(ir)

    getelementptr_type_dict = generate_getelementptr_type_dict(ir)

    # handle insertvalue with opaque ptrs
    ir, struct_type_dict = convert_opaque_ptrs_insertvalue(ir, getelementptr_type_dict)

    # handle phi with opaque ptrs
    # TODO this only works for phi ops that use results of insertvalue as input values
    # the types of these input values are stored in struct_type_dict
    ir, struct_type_dict = convert_opaque_ptrs_phi(ir, struct_type_dict)

    # handle extractvalue with opaque ptrs
    ir, struct_type_dict = convert_opaque_ptrs_extractvalue(ir, struct_type_dict)

    # handle ptrtoint with opaque ptrs
    # TODO this only works for ptrtoint where the ptr is result of getelementptr
    ir = convert_opaque_ptrs_ptrtoint(ir, getelementptr_type_dict)

    # need to convert inttoptr last
    # this is special case since ptr type is only known after the inttoptr operation
    # usually getelementptr is called on result of inttoptr
    ir = convert_opaque_ptrs_inttoptr(ir)

    # handle cmpxchg/atomicrmw with opaque ptrs
    ir = convert_opaque_ptrs_cmpxchg(ir, getelementptr_type_dict)
    ir = convert_opaque_ptrs_atomicrmw(ir, getelementptr_type_dict)

    # cast global smem when storing/loading
    ir = insert_bitcast_for_global_smem_load_store(ir)

    # need to do this after converting opaque ptrs for insertvalue/phi/extractvalue
    ir = insert_bitcast_for_vecs(ir, getelementptr_type_dict, struct_type_dict)

    # TODO handle cases when loading vec from ptr that is not result of getelementptr?
    ir = modify_func_signature(ir, ptr_type_dict)
    ir = rewrite_metadata(ir)

    return ir

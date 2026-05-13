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


def rewrite_simdgroup_event_type(ir: str) -> str:
    """
    Add %struct._simdgroup_event_t = type opaque
    Rewrite air.simdgroup_async_copy_2d return type from ptr to %struct._simdgroup_event_t*

    Before:
        declare ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(...)
        %r = call ptr @air.simdgroup_async_copy_2d.p3i8.p1i8(...)

    After:
        %struct._simdgroup_event_t = type opaque
        declare %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(...)
        %r = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d.p3i8.p1i8(...)
    """
    func_name = "air.simdgroup_async_copy_2d.p3i8.p1i8"
    if f"@{func_name}" not in ir:
        return ir

    event_ty = "%struct._simdgroup_event_t"
    event_ptr_ty = f"{event_ty}*"

    # add opaque struct type declaration before first define/declare
    if event_ty not in ir:
        first_decl = re.search(r"^(declare|define)\b", ir, re.MULTILINE)
        if first_decl:
            insert_pos = first_decl.start()
            ir = ir[:insert_pos] + f"{event_ty} = type opaque\n\n" + ir[insert_pos:]

    # rewrite declaration: ptr @air.simdgroup_async_copy_2d... -> event*
    ir = re.sub(
        rf"\bptr\s+(@{re.escape(func_name)}\b)",
        lambda m: f"{event_ptr_ty} {m.group(1)}",
        ir,
    )
    # rewrite call return type: call ptr @air.simdgroup_async_copy_2d... -> call event*
    ir = re.sub(
        rf"\bcall\s+ptr\s+(@{re.escape(func_name)}\b)",
        lambda m: f"call {event_ptr_ty} {m.group(1)}",
        ir,
    )

    # rewrite phi ptr that merges event values (from simdgroup 0 guard)
    # e.g. phi ptr [ null, %bb1 ], [ %event, %bb2 ] -> phi %struct._simdgroup_event_t* [...]
    # detect event vars (results of async copy calls)
    event_vars: set[str] = set()
    for m in re.finditer(rf"(%\w+)\s*=\s*call\s+{re.escape(event_ptr_ty)}", ir):
        event_vars.add(m.group(1))
    if event_vars:
        lines = ir.split("\n")
        new_lines = []
        for line in lines:
            phi_m = re.match(r"(\s*%\w+\s*=\s*)phi\s+ptr\s+(.*)", line)
            if phi_m:
                incoming = re.findall(r"\[\s*(%\w+|null)\s*,", phi_m.group(2))
                if any(v in event_vars for v in incoming):
                    line = f"{phi_m.group(1)}phi {event_ptr_ty} {phi_m.group(2)}"
            new_lines.append(line)
        ir = "\n".join(new_lines)

    return ir


def rewrite_simdgroup_wait_ptrs(ir: str) -> str:
    """
    Rewrite opaque ptrs in event alloca / wait pattern emitted by SimdgroupWaitOpToLLVM

    Run after rewrite_simdgroup_event_type so async copy results are already typed as %struct._simdgroup_event_t*

    Before:
        %event_alloca = alloca [2 x ptr], align 8
        %slot0 = getelementptr [2 x ptr], ptr %event_alloca, i64 0, i64 0
        %slot1 = getelementptr [2 x ptr], ptr %event_alloca, i64 0, i64 1
        %ev0   = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d...
        store ptr %ev0, ptr %slot0
        %ev1   = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d...
        store ptr %ev1, ptr %slot1
        %slot0w = getelementptr [2 x ptr], ptr %event_alloca, i64 0, i64 0
        call void @air.wait_simdgroup_events(i32 2, ptr %slot0w)
        declare void @air.wait_simdgroup_events(i32, ptr)

    After:
        %event_alloca = alloca [2 x %struct._simdgroup_event_t*], align 8
        %slot0 = getelementptr inbounds [2 x %struct._simdgroup_event_t*], [2 x %struct._simdgroup_event_t*]* %event_alloca, i64 0, i64 0
        %slot1 = getelementptr inbounds [2 x %struct._simdgroup_event_t*], [2 x %struct._simdgroup_event_t*]* %event_alloca, i64 0, i64 1
        %ev0   = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d...
        store %struct._simdgroup_event_t* %ev0, %struct._simdgroup_event_t** %slot0
        %ev1   = call %struct._simdgroup_event_t* @air.simdgroup_async_copy_2d...
        store %struct._simdgroup_event_t* %ev1, %struct._simdgroup_event_t** %slot1
        %slot0w = getelementptr inbounds [2 x %struct._simdgroup_event_t*], [2 x %struct._simdgroup_event_t*]* %event_alloca, i64 0, i64 0
        call void @air.wait_simdgroup_events(i32 2, %struct._simdgroup_event_t** %slot0w)
        declare void @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t**)
    """
    if "@air.wait_simdgroup_events" not in ir:
        return ir

    event_ptr_ty = "%struct._simdgroup_event_t*"
    event_ptr_ptr_ty = "%struct._simdgroup_event_t**"

    # rewrite declaration
    ir = re.sub(
        r"(declare\s+void\s+@air\.wait_simdgroup_events\s*\()i32,\s+ptr(\))",
        lambda m: f"{m.group(1)}i32, {event_ptr_ptr_ty}{m.group(2)}",
        ir,
    )

    lines = ir.split("\n")

    # collect vars that are %struct._simdgroup_event_t* (async copy results or phi merges)
    event_vars: set[str] = set()
    for line in lines:
        m = re.search(r"(%\w+)\s*=\s*call\s+%struct\._simdgroup_event_t\*", line)
        if m:
            event_vars.add(m.group(1))
        m = re.search(r"(%\w+)\s*=\s*phi\s+%struct\._simdgroup_event_t\*", line)
        if m:
            event_vars.add(m.group(1))

    # find alloca [N x ptr] vars
    alloca_vars: dict[str, int] = {}  # var -> N
    for line in lines:
        m = re.search(r"(%\w+)\s*=\s*alloca\s+\[(\d+)\s+x\s+ptr\]", line)
        if m:
            alloca_vars[m.group(1)] = int(m.group(2))

    # find geps from those alloca vars
    gep_source: dict[str, str] = {}  # gep_var -> alloca_var
    for line in lines:
        m = re.search(
            r"(%\w+)\s*=\s*getelementptr\s+(?:inbounds\s+)?\[(\d+)\s+x\s+ptr\],\s+ptr\s+(%\w+),\s+i\d+\s+0,\s+i\d+\s+\d+",
            line,
        )
        if m and m.group(3) in alloca_vars:
            gep_source[m.group(1)] = m.group(3)

    # find event allocas via the wait call argument
    event_alloca_vars: set[str] = set()
    for line in lines:
        m = re.search(
            r"call\s+void\s+@air\.wait_simdgroup_events\s*\([^,]+,\s+ptr\s+(%\w+)",
            line,
        )
        if m and m.group(1) in gep_source:
            event_alloca_vars.add(gep_source[m.group(1)])

    if not event_alloca_vars:
        return ir

    event_gep_vars: set[str] = {v for v, a in gep_source.items() if a in event_alloca_vars}

    # rewrite alloca/gep/store/wait call
    new_lines = []
    for line in lines:
        # alloca [N x ptr] -> [N x event_ptr_ty] for event allocas
        def rewrite_alloca(m):
            var, n, rest = m.group(1), m.group(2), m.group(3)
            if var in event_alloca_vars:
                return f"{var} = alloca [{n} x {event_ptr_ty}]{rest}"
            return m.group(0)

        line = re.sub(r"(%\w+)\s*=\s*alloca\s+\[(\d+)\s+x\s+ptr\](.*)", rewrite_alloca, line)

        # getelementptr [N x ptr], ptr %alloca -> typed gep for event allocas
        def rewrite_gep(m):
            gep_var, n, alloca_var, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            if alloca_var in event_alloca_vars:
                arr_ty = f"[{n} x {event_ptr_ty}]"
                return f"{gep_var} = getelementptr inbounds {arr_ty}, {arr_ty}* {alloca_var}{rest}"
            return m.group(0)

        line = re.sub(
            r"(%\w+)\s*=\s*getelementptr\s+(?:inbounds\s+)?\[(\d+)\s+x\s+ptr\],\s+ptr\s+(%\w+)(.*)",
            rewrite_gep,
            line,
        )

        # store ptr %event, ptr %slot -> typed store
        def rewrite_store(m):
            prefix, val_var, dst_var, rest = m.group(1), m.group(2), m.group(3), m.group(4)
            if val_var in event_vars and dst_var in event_gep_vars:
                return f"{prefix}store {event_ptr_ty} {val_var}, {event_ptr_ptr_ty} {dst_var}{rest}"
            return m.group(0)

        line = re.sub(
            r"(\s*)store\s+ptr\s+(%\w+),\s+ptr\s+(%\w+)((?:,\s*align\s+\d+)?[^\n]*)",
            rewrite_store,
            line,
        )

        # call void @air.wait_simdgroup_events(i32 N, ptr %slot0)
        def rewrite_wait(m):
            n_arg, slot_var = m.group(1), m.group(2)
            if slot_var in event_gep_vars:
                return f"call void @air.wait_simdgroup_events(i32 {n_arg}, {event_ptr_ptr_ty} {slot_var})"
            return m.group(0)

        line = re.sub(
            r"call\s+void\s+@air\.wait_simdgroup_events\s*\(i32\s+(\w+),\s+ptr\s+(%\w+)\)",
            rewrite_wait,
            line,
        )

        new_lines.append(line)

    return "\n".join(new_lines)


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


def rewrite_splat_vector_constants(ir: str) -> str:
    """
    Expand splat vector constants to full form

    MLIR emits dense splat vectors as <T val> shorthand, but metal LLVM requires all elements to be listed

    Before: <2 x i64> <i64 8>
    After:  <2 x i64> <i64 8, i64 8>
    """

    def expand(m):
        n = int(m.group(1))
        elem_ty = m.group(2)
        elem_val = m.group(3).strip()
        if n <= 1:
            return m.group(0)
        return f"<{n} x {elem_ty}> <{', '.join([elem_val] * n)}>"

    # match <N x T> <T val> where the inner braces contain single element (no comma)
    return re.sub(r"<(\d+)\s+x\s+(\w+)>\s+<(\w+\s+[^,>]+)>", expand, ir)


def rewrite_local_ptrs(ir: str) -> str:
    """
    Rewrite bare ptr (addrspace 0, stack allocations) to typed ptrs in gep, load, store instrs

    Before:
        %21 = getelementptr [1 x %"struct.metal::simdgroup_matrix"], ptr %19, i32 0, i32 0, i32 0
        store <64 x float> %20, ptr %21, align 256
        %201 = load <64 x float>, ptr %21, align 256
    After:
        %21 = getelementptr [1 x %"struct.metal::simdgroup_matrix"], [1 x %"struct.metal::simdgroup_matrix"]* %19, i32 0, i32 0, i32 0
        store <64 x float> %20, <64 x float>* %21, align 256
        %201 = load <64 x float>, <64 x float>* %21, align 256
    """
    # match arrays ([N x T]), vectors (<N x T>), quoted names (%"..."),
    # plain named types (%name), or primitive keywords (float, i32, etc.)
    _T = r'(?:\[[^\]]+\]|<[^>]+>|%"[^"]+"|%\w+|\w+)'

    # getelementptr ELEM_TYPE, ptr %var -> getelementptr ELEM_TYPE, ELEM_TYPE* %var
    ir = re.sub(
        rf"(getelementptr(?:\s+inbounds)?\s+)({_T}),\s+ptr\s+(?=%)",
        lambda m: f"{m.group(1)}{m.group(2)}, {m.group(2)}* ",
        ir,
    )

    # load ELEM_TYPE, ptr %var -> load ELEM_TYPE, ELEM_TYPE* %var
    ir = re.sub(
        rf"(load\s+({_T})),\s+ptr\s+(?=%)",
        lambda m: f"{m.group(1)}, {m.group(2)}* ",
        ir,
    )

    # store ELEM_TYPE %val, ptr %var -> store ELEM_TYPE %val, ELEM_TYPE* %var
    ir = re.sub(
        rf"(store\s+({_T})\s+[^,\n]+),\s+ptr\s+(?=%)",
        lambda m: f"{m.group(1)}, {m.group(2)}* ",
        ir,
    )

    return ir


def rewrite_air_simdgroup_async_copy_ptrs(ir: str) -> str:
    """
    Rewrite opaque ptr addrspace args in air.simdgroup_async_copy_2d declaration and calls

    Ptr types in function name suffix:
      .p<dst_as><dst_elem>.p<src_as><src_elem>
    e.g. .p3i8.p1i8 -> dst: i8 addrspace(3)*, src: i8 addrspace(1)*

    Before:
        declare ... @air.simdgroup_async_copy_2d.p3i8.p1i8(
            i64, i64, ptr addrspace(3), i64, i64, <2 x i64>,
            ptr addrspace(1), i64, i64, <2 x i64>, <2 x i64>, i32)
        call ... @air.simdgroup_async_copy_2d.p3i8.p1i8(
            ..., ptr addrspace(3) %smem, ..., ptr addrspace(1) %glb, ...)
    After:
        declare ... @air.simdgroup_async_copy_2d.p3i8.p1i8(
            i64, i64, i8 addrspace(3)*, i64, i64, <2 x i64>,
            i8 addrspace(1)*, i64, i64, <2 x i64>, <2 x i64>, i32)
        call ... @air.simdgroup_async_copy_2d.p3i8.p1i8(
            ..., i8 addrspace(3)* %smem, ..., i8 addrspace(1)* %glb, ...)
    """
    func_pat = re.compile(r"@air\.simdgroup_async_copy_2d\.p(\d+)(\w+)\.p(\d+)(\w+)")

    new_lines = []
    for line in ir.split("\n"):
        m = func_pat.search(line)
        if not m:
            new_lines.append(line)
            continue

        dst_as = m.group(1)
        dst_ty = _AIR_ELEM_TYPE_MAP.get(m.group(2), m.group(2))
        src_as = m.group(3)
        src_ty = _AIR_ELEM_TYPE_MAP.get(m.group(4), m.group(4))

        # replace ptr addrspace(dst) -> dst_ty addrspace(dst)*
        line = re.sub(
            rf"\bptr\s+addrspace\({re.escape(dst_as)}\)",
            f"{dst_ty} addrspace({dst_as})*",
            line,
        )
        # replace ptr addrspace(src) -> src_ty addrspace(src)*
        line = re.sub(
            rf"\bptr\s+addrspace\({re.escape(src_as)}\)",
            f"{src_ty} addrspace({src_as})*",
            line,
        )
        new_lines.append(line)

    return "\n".join(new_lines)


def rewrite_air_simdgroup_store_ptrs(ir: str) -> str:
    """
    Rewrite opaque ptr addrspace args in air.simdgroup_matrix_8x8_store declaration and calls

    Element type comes from function name suffix (e.g. .p3f32 -> float)

    Before:
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
            <64 x float>, ptr addrspace(1), i64, <2 x i64>, i1)
        call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
            <64 x float> %v, ptr addrspace(1) %p, i64 %s, <2 x i64> %c, i1 false)
    After:
        declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
            <64 x float>, float addrspace(1)*, i64, <2 x i64>, i1)
        call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(
            <64 x float> %v, float addrspace(1)* %p, i64 %s, <2 x i64> %c, i1 false)
    """
    # collect store function names and element types from suffix
    store_elem: dict[str, str] = {}  # func_name -> llvm elem type
    for m in re.finditer(
        r"@(air\.simdgroup_matrix_8x8_store\.[^()\s]+)",
        ir,
    ):
        func_name = m.group(1).strip()
        if func_name in store_elem:
            continue
        # ptr suffix is the last .p<N><elem> segment
        ptr_m = re.search(r"\.p\d+(\w+)$", func_name)
        if not ptr_m:
            continue
        llvm_type = _AIR_ELEM_TYPE_MAP.get(ptr_m.group(1))
        if llvm_type:
            store_elem[func_name] = llvm_type

    if not store_elem:
        return ir

    new_lines = []
    for line in ir.split("\n"):
        for func_name, elem_type in store_elem.items():
            if f"@{func_name}" not in line:
                continue
            # replace ptr addrspace(N) with elem_type addrspace(N)* for any N
            line = re.sub(
                r"\bptr\s+addrspace\((\d+)\)",
                lambda mo, et=elem_type: f"{et} addrspace({mo.group(1)})*",
                line,
            )
        new_lines.append(line)

    return "\n".join(new_lines)


def rewrite_metadata(ir: str) -> str:
    # rewrite metadata func ptr reference
    # e.g. ptr @funcname -> void (arg types)* @funcname
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


def fix_async_copy_ptr_type_mismatches(ir: str) -> str:
    """
    Insert bitcasts before async copy calls where ptr arg's type differs from the call-site annotation

    After typed-ptr rewrites, a var like %173 may be defined as
    half addrspace(1)* but annotated in async copy call as i8 addrspace(1)*.
    Metal's LLVM rejects this mismatch.

    Before:
        half addrspace(1)* %173 (definition)
        call ... @air.simdgroup_async_copy_2d...(... i8 addrspace(1)* %173, ...)
    After:
        half addrspace(1)* %173 (definition)
        %async_copy_ptr_cast_0 = bitcast half addrspace(1)* %173 to i8 addrspace(1)*
        call ... @air.simdgroup_async_copy_2d...(... i8 addrspace(1)* %async_copy_ptr_cast_0, ...)
    """
    # var -> actual typed ptr type from definition sites
    var_type: dict[str, str] = {}

    for line in ir.split("\n"):
        # %var = ... to ELEM addrspace(N)* (bitcast, inttoptr)
        m = re.match(r"\s*(%\w+)\s*=\s*\S.*?\bto\s+(\w+\s+addrspace\(\d+\)\*)\s*$", line)
        if m:
            var_type[m.group(1)] = m.group(2)
            continue

        # %var = getelementptr [inbounds] ELEM, ELEM addrspace(N)* %base, ...
        m = re.match(
            r"\s*(%\w+)\s*=\s*getelementptr\b[^,\n]*,\s+(<\d+\s+x\s+\w+>|\w+)\s+addrspace\((\d+)\)\*",
            line,
        )
        if m:
            var_type[m.group(1)] = f"{m.group(2)} addrspace({m.group(3)})*"
            continue

        # %var = extractvalue { ..., ELEM addrspace(N)*, ... } %agg, IDX
        m = re.match(r"\s*(%\w+)\s*=\s*extractvalue\s+(\{[^}]+\})\s+\S+,\s*(\d+)", line)
        if m:
            fields = [f.strip() for f in m.group(2).strip("{}").split(",")]
            idx = int(m.group(3))
            if idx < len(fields):
                pm = re.search(r"(\w+\s+addrspace\(\d+\)\*)", fields[idx])
                if pm:
                    var_type[m.group(1)] = pm.group(1)
            continue

        # ELEM addrspace(N)* [attrs] %var (function args, phi, etc.)
        for m in re.finditer(r"(\w+)\s+addrspace\((\d+)\)\*(?:\s+\w+)*\s+(%\w+)(?=\s*[,){\[]|\s*$)", line):
            if m.group(3) not in var_type:
                var_type[m.group(3)] = f"{m.group(1)} addrspace({m.group(2)})*"

    func_pat = re.compile(r"@air\.simdgroup_async_copy_2d\.[^(]+")
    cast_idx = 0
    new_lines = []

    for line in ir.split("\n"):
        if not func_pat.search(line):
            new_lines.append(line)
            continue

        indent = re.match(r"(\s*)", line).group(1)

        # for each typed ptr arg, insert bitcast if var's definition type differs
        changed = True
        while changed:
            changed = False
            for arg_m in re.finditer(r"(\w+)\s+addrspace\((\d+)\)\*\s+(%\w+)", line):
                annotated_ptr = f"{arg_m.group(1)} addrspace({arg_m.group(2)})*"
                var = arg_m.group(3)
                actual_type = var_type.get(var)
                if actual_type and actual_type != annotated_ptr:
                    cast_var = f"%async_copy_ptr_cast_{cast_idx}"
                    cast_idx += 1
                    new_lines.append(f"{indent}{cast_var} = bitcast {actual_type} {var} to {annotated_ptr}")
                    line = line.replace(f"{annotated_ptr} {var}", f"{annotated_ptr} {cast_var}", 1)
                    changed = True
                    break  # restart finditer on the updated line

        new_lines.append(line)

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

    # expand single-element splat vector constants (e.g. <2 x i64> <i64 8> -> <i64 8, i64 8>)
    ir = rewrite_splat_vector_constants(ir)

    # rewrite opaque ptrs in air.simdgroup_matrix_8x8_* load declarations
    ir = rewrite_air_simdgroup_decl_ptrs(ir)

    # rewrite ptr addrspace args in store declarations/calls (addrspace from call
    # site, elem type from function name suffix)
    ir = rewrite_air_simdgroup_store_ptrs(ir)

    # rewrite ptr addrspace args in async copy declarations/calls (both dst and
    # src ptr types derived from function name suffix)
    ir = rewrite_air_simdgroup_async_copy_ptrs(ir)

    # add %struct._simdgroup_event_t type and rewrite async copy return type
    ir = rewrite_simdgroup_event_type(ir)

    # rewrite event alloca, gep, stores, and wait call to use typed ptrs
    ir = rewrite_simdgroup_wait_ptrs(ir)

    # rewrite ptrs to include types
    ir = rewrite_ptrs(ir)

    # rewrite bare ptr (addrspace 0) in GEP/load/store from local allocas
    ir = rewrite_local_ptrs(ir)

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

    # after all typed ptr rewrites, fix async copy calls where a var's
    # defined type differs from the callsite annotation (e.g. half* vs i8*)
    ir = fix_async_copy_ptr_type_mismatches(ir)

    return ir

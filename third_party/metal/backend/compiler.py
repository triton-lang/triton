from loguru import logger
import tempfile
import subprocess
import os
import re
import shutil
import sys
from typing import Optional, Tuple, Dict, Any, Union

class MetalCompiler:
    """
    MetalCompiler provides compilation routines for Metal backend.
    """

    def compile(self, source: str, options: dict, reflection: bool = False) -> Union[bytes, Tuple[bytes, dict]]:
        """
        Compile Metal source code to binary.

        Args:
            source (str): Metal shader source code.
            options (dict): Compilation options.
            reflection (bool): If True, also parse and return lightweight reflection metadata.

        Returns:
            bytes or (bytes, dict): Compiled binary suitable for MetalDriver.load_binary.
                                  When reflection=True returns (binary_bytes, metadata).

        Raises:
            RuntimeError: If compilation or linking fails.
        """
        air_path, metallib_path = None, None
        src_path = ''
        # Ensure xcrun is available before proceeding
        if shutil.which("xcrun") is None:
            raise RuntimeError("Required tool 'xcrun' not found in PATH; cannot invoke Metal compiler/linker.")
        try:
            # Write source to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as src_file:
                src_file.write(source.encode("utf-8"))
                src_path = src_file.name
        except Exception as e:
            # Fail-fast: cannot continue if temp file creation fails.
            logger.error(f"Failed to create temp file: {e}")
            raise RuntimeError(f"Failed to create temporary source file for Metal compilation: {e}") from e

        try:
            # Compile .metal to .air
            air_path = src_path.replace(".metal", ".air")
            metal_cmd = ["xcrun", "metal", src_path, "-o", air_path]
            if options.get("macros"):
                for macro in options["macros"]:
                    metal_cmd += ["-D", macro]
            _run_metal_compile(metal_cmd, "Metal compilation failed")

            # Link .air to .metallib
            metallib_path = src_path.replace(".metal", ".metallib")
            metallib_cmd = ["xcrun", "metallib", air_path, "-o", metallib_path]
            _run_metal_compile(metallib_cmd, "Metal linking failed")

            # Read binary output
            with open(metallib_path, "rb") as f:
                binary = f.read()

            # If reflection requested generate best-effort structured metadata by parsing
            # kernel declarations in the source. This is intentionally conservative —
            # it attempts to extract kernel names and per-argument information such
            # as the raw parameter text, a cleaned type, parameter name and any
            # Metal buffer index annotations ( [[ buffer(N) ]] ), plus an inferred
            # address-space (device/constant/threadgroup/stack).
            metadata = {}
            if reflection:
                try:
                    # Find kernel declarations. We cannot rely on a single simple regex
                    # to capture the full parameter list when that list may contain nested
                    # parentheses (e.g. attributes like [[ buffer(0) ]]). Instead locate
                    # the kernel name with a regex and then scan for the matching closing
                    # parenthesis to extract the full parameter text.
                    name_pattern = re.compile(r'\bkernel\b(?:\s+\w+)?\s+([A-Za-z_]\w*)\s*\(', re.MULTILINE)
                    # Conservative parsing: find any [[ ... ]] attributes and inspect their
                    # contents for a "buffer(N)" specification. This handles spacing variants
                    # such as "[[ buffer(0) ]]" or "[[buffer(0)]]" and multiple attributes.
                    attr_inner_pattern = re.compile(r'\[\[(.*?)\]\]')
                    buffer_inner_pattern = re.compile(r'buffer\s*\(\s*(\d+)\s*\)')
                    kernels = []
                    for m in name_pattern.finditer(source):
                        kname = m.group(1)
                        # scan for the matching closing parenthesis after m.end()
                        start = m.end()  # position after the '('
                        depth = 1
                        i = start
                        end = None
                        while i < len(source):
                            ch = source[i]
                            if ch == '(':
                                depth += 1
                            elif ch == ')':
                                depth -= 1
                                if depth == 0:
                                    end = i
                                    break
                            i += 1
                        params = ''
                        if end is not None:
                            params = source[start:end].strip()
                        arg_list = []
                        if params:
                            raw_args = [p.strip() for p in params.split(',') if p.strip()]
                            for raw in raw_args:
                                # Find buffer(...) either directly in the token or inside bracketed attributes.
                                # Some compilers/formatters may emit attributes with varying spacing, e.g.:
                                #   [[buffer(0)]]  or  [[  buffer(0)  ]]
                                # Try a direct search first (covers most cases), then fall back to scanning attributes.
                                buffer_index = None
                                bmatch = buffer_inner_pattern.search(raw)
                                if bmatch:
                                    try:
                                        buffer_index = int(bmatch.group(1))
                                    except (ValueError, TypeError):
                                        buffer_index = None
                                else:
                                    for attr_match in attr_inner_pattern.findall(raw):
                                        inner = attr_match.strip()
                                        bmatch = buffer_inner_pattern.search(inner)
                                        if bmatch:
                                            try:
                                                buffer_index = int(bmatch.group(1))
                                            except (ValueError, TypeError):
                                                # Keep conservative behaviour: ignore invalid values
                                                buffer_index = None
                                            break
                                # Debugging: log the raw token and discovered buffer_index (if any).
                                # This log is intentionally low-volume and safe for test-time diagnostics.
                                logger.debug(f"MetalCompiler: parsed arg raw={raw!r} buffer_index={buffer_index}")
                                # Remove attribute annotations like [[ buffer(0) ]] or [[ ... ]]
                                cleaned = re.sub(r'\[\[.*?\]\]', '', raw).strip()
                                # Normalize whitespace
                                cleaned = re.sub(r'\s+', ' ', cleaned)
                                # Split tokens to separate type and name. Typical form:
                                #   "device float *in"  -> type="device float *", name="in"
                                #   "uint count" -> type="uint", name="count"
                                tokens = cleaned.split(' ')
                                if len(tokens) >= 2:
                                    param_name = tokens[-1]
                                    param_type = ' '.join(tokens[:-1])
                                elif len(tokens) == 1:
                                    # No explicit name, treat whole token as type
                                    param_name = None
                                    param_type = tokens[0]
                                else:
                                    param_name = None
                                    param_type = None
                                # Infer address space if present in param_type
                                addr = None
                                if param_type:
                                    if 'device' in param_type:
                                        addr = 'device'
                                    elif 'threadgroup' in param_type:
                                        addr = 'threadgroup'
                                    elif 'constant' in param_type:
                                        addr = 'constant'
                                # Validate buffer_index type if present
                                if buffer_index is not None:
                                    assert isinstance(buffer_index, int)
                                arg_list.append({
                                    'raw': raw,
                                    'type': param_type,
                                    'name': param_name,
                                    'buffer_index': buffer_index,
                                    'address_space': addr,
                                })
                        kernels.append({'name': kname, 'args': arg_list})
                    metadata['kernels'] = kernels
                except Exception as e:
                    logger.warning(f"Reflection parsing failed: {e}")
            if reflection:
                return binary, metadata
            return binary

        finally:
            # Clean up temp files
            for path in [air_path, metallib_path, src_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {path}: {e}")

def _run_metal_compile(cmd, error_msg):
    """
    Helper to run metal/metallib subprocess and handle errors.
    """
    logger.info(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        out = proc.stdout.decode('utf-8', errors='replace')
        err = proc.stderr.decode('utf-8', errors='replace')
        logger.error(f"{error_msg}: stdout={out} stderr={err}")
        raise RuntimeError(f"{error_msg}: stdout={out} stderr={err}")
"""
Structured logging for Triton compilation and execution.

This module provides utilities for structured logging of Triton operations,
particularly focused on kernel compilation tracing. It includes custom log
formatters, handlers, and helper functions to capture and store detailed
information about Triton kernels, their compilation artifacts, and execution.
"""

import atexit
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, cast, Dict, Optional, Union

from .. import knobs


# Define a custom LogRecord type with additional attributes for structured logging
class TritonLogRecord(logging.LogRecord):
    metadata: Dict[str, Any]
    payload: Optional[Union[str, Dict[str, Any], list]]


TEXT_FILE_EXTENSIONS = [".ttir", ".ttgir", ".llir", ".ptx", ".amdgcn", ".json"]
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit for file content extraction

log = logging.getLogger(__name__)
triton_trace_log = logging.getLogger("triton.trace")
triton_trace_folder = knobs.structured_logging.triton_trace
TRITON_TRACE_HANDLER = None


def _init_logs():
    """
    Initialize the logging system for Triton tracing.
    
    Sets up the global trace handler and configures the logger with appropriate
    formatter and log level.
    """
    global TRITON_TRACE_HANDLER
    global triton_trace_folder
    if TRITON_TRACE_HANDLER is None:
        TRITON_TRACE_HANDLER = TritonTraceHandler(triton_trace_folder)
        triton_trace_log.setLevel(logging.DEBUG)
        TRITON_TRACE_HANDLER.setFormatter(TritonJsonFormatter())
        triton_trace_log.addHandler(TRITON_TRACE_HANDLER)


def get_stack_trace(skip=1):
    """
    Get call stack trace for the current execution context.

    Extracts stack trace information using torch's CapturedTraceback utility,
    providing detailed information about each frame in the call stack.

    Args:
        skip (int): Number of frames to skip from the start of the stack

    Returns:
        List[Dict]: List of frame information dictionaries containing line numbers,
                   function names, filenames, and code snippets
    """
    frames = []
    try:
        from torch.utils._traceback import CapturedTraceback
    except ImportError:
        return []

    for frame in CapturedTraceback.extract(skip=skip).summary():
        frames.append(
            {
                "line": frame.lineno,
                "name": frame.name,
                "filename": frame.filename,
                "loc": frame.line,
            }
        )
    return frames


class TritonJsonFormatter(logging.Formatter):
    """
    Format log records as JSON for Triton compilation tracing.
    
    This formatter converts log records with metadata and payload into NDJSON format,
    suitable for structured logging and later analysis. It handles special attributes
    added by the Triton tracing system, such as metadata dictionaries and payload data.
    """

    def format(self, record: logging.LogRecord):
        # Cast to our custom type for type checking
        # Using typing.cast to tell the type checker that record is actually a TritonLogRecord
        # This doesn't change the runtime behavior, just helps with type checking
        record_as_triton = cast(TritonLogRecord, record)

        log_entry = getattr(record_as_triton, "metadata", {})
        log_entry["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")
        payload = getattr(record, "payload", None)
        if payload is not None:
            log_entry["payload"] = json.loads(payload)
        # NDJSON format requires a newline at the end of each line
        return json.dumps(log_entry, separators=(",", ":")) + "\n"


class TritonTraceHandler(logging.StreamHandler):
    """
    A handler for Triton compilation tracing that outputs NDJSON files.
    
    This handler creates and manages log files for Triton kernel compilation traces.
    It supports creating new files for different compilation events and handles
    proper cleanup of file resources. When running in a distributed environment,
    it automatically adds rank information to filenames.
    """

    def __init__(self, root_dir: Optional[str], prefix="dedicated_log_triton_trace_"):
        logging.Handler.__init__(self)
        self.root_dir = root_dir
        self.prefix = prefix
        self.stream = None
        self.first_record = True
        # Register cleanup handler for program exit
        atexit.register(self._cleanup)

    def emit(self, record):
        if self.stream is None:
            if self.root_dir is not None:
                os.makedirs(self.root_dir, exist_ok=True)
                ranksuffix = ""
                try: 
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        ranksuffix = f"rank_{dist.get_rank()}_"
                except ImportError:
                    pass
                filename = f"{self.prefix}{ranksuffix}"
                self.stream = open(
                    os.path.join(self.root_dir, f"{filename}.ndjson"),
                    mode="a+",
                )
                log.debug("TritonTraceHandler: logging to %s", self.stream.name)
            else:
                triton_trace_log.removeHandler(self)
                return
        if self.stream:
            formatted = self.format(record)
            self.stream.write(formatted)
            self.flush()

    def close(self):
        """Close the current file."""
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Solution adopted from PyTorch PR #120289
                logging.StreamHandler.close(self)
        finally:
            self.release()

    def _cleanup(self):
        """Ensure proper cleanup on program exit"""
        if self.stream is not None:
            self.close()


def trace_structured_triton(
    name: str,
    metadata_fn: Callable[[], Dict[str, Any]] = dict,
    *,
    payload_fn: Callable[[], Optional[Union[str, object]]] = lambda: None,
    new_file: bool = True,
):
    """
    Record structured trace information for Triton kernel compilation.
    
    This function is the main entry point for logging structured trace events
    in the Triton system. It handles initialization of the logging system if needed,
    creates new log files when requested, and formats the trace data with metadata
    and payload information.

    Args:
        name (str): Name of the trace event (e.g., "compilation", "execution")
        metadata_fn (Callable): Function that returns a dictionary of metadata to include
                               in the trace record
        payload_fn (Callable): Function that returns the payload data (can be a string,
                              dictionary, or other serializable object)
        new_file (bool): Whether to create a new log file for this trace event
    """
    global TRITON_TRACE_HANDLER
    global triton_trace_folder
    # Initialize logging if needed
    if not triton_trace_log.handlers or TRITON_TRACE_HANDLER is None:
        # Clear existing handlers
        for handler in list(triton_trace_log.handlers):
            triton_trace_log.removeHandler(handler)

        TRITON_TRACE_HANDLER = None
        _init_logs()

    # Create new file if requested
    if new_file and TRITON_TRACE_HANDLER and TRITON_TRACE_HANDLER.stream is not None:
        TRITON_TRACE_HANDLER.close()
        TRITON_TRACE_HANDLER.stream = None

    metadata_dict: Dict[str, Any] = {"event_type": name}
    metadata_dict["pid"] = os.getpid()
    custom_metadata = metadata_fn()
    if custom_metadata:
        metadata_dict.update(custom_metadata)

    metadata_dict["stack"] = get_stack_trace()

    # Log the record
    payload = payload_fn()
    triton_trace_log.debug("", extra={"metadata": metadata_dict, "payload": payload})


def extract_python_source_info(trace_data: Dict[str, Any], source: Any):
    """
    Extract Python source code information from the source object and add it to trace_data.
    
    This function uses Python's inspect module to extract source code information
    from the provided source object (typically an ASTSource or IRSource instance).
    It adds file path, line numbers, and the actual source code to the trace_data.

    Args:
        trace_data (Dict[str, Any]): Dictionary to store extracted information
        source (Union[ASTSource, IRSource]): Source object containing kernel function information
    """

    import inspect

    # Get the original Python source code for the kernel
    target_fn = source.fn.fn
    python_source_file = inspect.getfile(target_fn)
    source_lines, start_line_number = inspect.getsourcelines(target_fn)
    end_line_number = start_line_number + len(source_lines)

    trace_data["python_source"] = {
        "file_path": python_source_file,
        "start_line": start_line_number,
        "end_line": end_line_number,
        "code": inspect.getsource(target_fn),
    }


def extract_file_content(trace_data: Dict[str, Any], metadata_group: Dict[str, str]):
    """
    Extract file content from metadata_group and add it to trace_data.

    Args:
        trace_data (Dict): Dictionary to store extracted information
        metadata_group (Dict): Dictionary mapping filenames to file paths
    """
    for ir_filename, file_path in metadata_group.items():
        # Add file path to trace data
        trace_data["file_path"][ir_filename] = file_path

        # Check if this is a text file we can read
        if any(ir_filename.endswith(ext) for ext in TEXT_FILE_EXTENSIONS):
            try:
                # Check file size before reading to avoid memory issues
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    trace_data["file_content"][
                        ir_filename
                    ] = f"<file too large: {file_size} bytes>"
                    continue

                with open(file_path, "r") as f:
                    trace_data["file_content"][ir_filename] = f.read()
            except (IOError, UnicodeDecodeError):
                # Skip if file can't be read as text
                pass


def maybe_trace_triton(
    metadata_path: Optional[Union[str, Path]],
    metadata_group: Dict[str, str],
    src: Optional[Union[Any, Any]] = None,
    event_type: str = "compilation",
):
    """
    Collect and trace Triton kernel compilation information for debugging and profiling.

    This function gathers metadata, IR files, and source code information about a Triton
    kernel compilation, then logs it through the tracing system if tracing is enabled.
    It collects information from multiple sources:
    1. JSON metadata file (if provided)
    2. PyTorch compilation context (if available)
    3. IR and other compilation artifact files
    4. Python source code of the kernel function

    Args:
        metadata_path (Optional[Union[str, Path]]): Path to the JSON metadata file for the compiled kernel
        metadata_group (Dict[str, str]): Dictionary mapping filenames to file paths for all compilation artifacts
        src (Optional[Union[ASTSource, IRSource]]): Source object containing kernel information
        event_type (str): Type of event being traced (default: "compilation")

    Returns:
        Dict[str, Any]: Dictionary containing all collected trace data, even if tracing is disabled
    """
    # Initialize a dictionary with defaultdict to avoid key errors
    trace_data = defaultdict(dict)

    # Extract metadata from the JSON file if available
    if metadata_path is not None:
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                # Copy all metadata fields to the trace data
                for key, value in metadata.items():
                    trace_data["metadata"][key] = value
        except (IOError, json.JSONDecodeError) as e:
            log.warning(f"Failed to load metadata from {metadata_path}: {e}")
    # Handle torch._guards which might not be recognized by type checker
    try:
        import torch
        trace_id = torch._guards.CompileContext.current_trace_id()  # type: ignore
    except (AttributeError, ImportError):
        trace_id = None
    cid = trace_id.compile_id if trace_id else None
    if cid is not None:
        if cid.compiled_autograd_id is not None:
            trace_data["pt_info"]["compiled_autograd_id"] = cid.compiled_autograd_id
        if cid.frame_id is not None:
            trace_data["pt_info"]["frame_id"] = cid.frame_id
        if cid.frame_compile_id is not None:
            trace_data["pt_info"]["frame_compile_id"] = cid.frame_compile_id
    if trace_id:
        trace_data["pt_info"]["attempt"] = trace_id.attempt
    # Extract content from all IR and other files in the metadata group
    if metadata_group:
        extract_file_content(trace_data, metadata_group)

    # Extract Python source code information if available
    if src:
        extract_python_source_info(trace_data, src)

    # Log the collected information through the tracing system
    trace_structured_triton(
        event_type,
        payload_fn=lambda: json.dumps(trace_data),
    )

    return trace_data


_init_logs()

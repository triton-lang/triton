import logging
import os
import json
import time
import atexit
from collections import defaultdict
from typing import Optional, Callable, Any, Union, Dict

# Constants for text file extensions
TEXT_FILE_EXTENSIONS = ['.ttir', '.ttgir', '.llir', '.ptx', '.amdgcn', '.json']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit for file content extraction

# Configure basic logging
log = logging.getLogger(__name__)
triton_trace_log = logging.getLogger("triton.trace")

# This environment variable is used to enable and specify the directory for tracing logs
triton_trace_folder = os.environ.get("TRITON_TRACE", None)

# This environment variable is used to disable file content extraction. If the following
# source code mapping process or file content extraction is not needed, this can be set
# to reduce log size.
triton_trace_no_file_content = os.environ.get("TRITON_TRACE_NO_FILE_CONTENT") == '1'

# Handler for tracing
TRITON_TRACE_HANDLER = None


def get_simplified_stack_trace(skip=1):
    """
    Get a simplified stack trace.

    Args:
        skip (int): Number of frames to skip from the start

    Returns:
        List[Dict]: List of frame information dictionaries
    """
    frames = []
    try:
        from torch.utils._traceback import CapturedTraceback
    except ImportError:
        return frames  # Return an empty list if import fails

    for frame in CapturedTraceback.extract(skip=skip).summary():
        frames.append({
            "line": frame.lineno,
            "name": frame.name,
            "filename": frame.filename,
            "loc": frame.line,
        })
    return frames


class TritonJsonFormatter(logging.Formatter):
    """Format log records as JSON for Triton compilation tracing."""

    def format(self, record):
        log_entry = record.metadata if hasattr(record, "metadata") else {}

        # Add timestamp
        log_entry["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Add payload if provided
        if hasattr(record, "payload") and record.payload is not None:
            try:
                if isinstance(record.payload, str) and (record.payload.startswith("{")
                                                        or record.payload.startswith("[")):
                    log_entry["payload"] = json.loads(record.payload)
                else:
                    log_entry["payload"] = record.payload
            except json.JSONDecodeError:
                # Keep payload as string if JSON parsing fails
                log_entry["payload"] = record.payload

        return json.dumps(log_entry, indent=2)


class TritonTraceHandler(logging.StreamHandler):
    """A handler for Triton compilation tracing that outputs JSON to separate files."""

    def __init__(self, root_dir: Optional[str], prefix="dedicated_log_triton_trace_"):
        logging.Handler.__init__(self)
        self.root_dir = root_dir
        self.prefix = prefix
        self.stream = None
        self.first_record = True
        # Register cleanup handler for program exit
        atexit.register(self._cleanup)

    def emit(self, record):
        os.makedirs(self.root_dir, exist_ok=True)

        # Close previous file if exists
        if self.stream is not None:
            self.close()

        # Create unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}{timestamp}.json"

        self.stream = open(os.path.join(self.root_dir, filename), "w")
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
                # PyTorch Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                # Also see PyTorch Issue #42378: we also rely on
                # self._closed being set to True there
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
    Record structured trace information for Triton kernel compilation

    Args:
        name: Name of the trace event
        metadata_fn: Function that returns metadata dictionary
        payload_fn: Function that returns payload data
        new_file: Whether to create a new log file for this call
    """
    global TRITON_TRACE_HANDLER
    global triton_trace_folder
    # Initialize logging if needed
    if not triton_trace_log.handlers or TRITON_TRACE_HANDLER is None:
        triton_trace_log.setLevel(logging.DEBUG)
        triton_trace_log.propagate = False

        # Clear existing handlers
        for handler in list(triton_trace_log.handlers):
            triton_trace_log.removeHandler(handler)

        TRITON_TRACE_HANDLER = TritonTraceHandler(triton_trace_folder)
        formatter = TritonJsonFormatter()
        TRITON_TRACE_HANDLER.setFormatter(formatter)
        triton_trace_log.addHandler(TRITON_TRACE_HANDLER)

    # Create new file if requested
    if new_file and TRITON_TRACE_HANDLER and TRITON_TRACE_HANDLER.stream is not None:
        TRITON_TRACE_HANDLER.close()
        TRITON_TRACE_HANDLER.stream = None

    record = {"event_type": name}
    record["pid"] = os.getpid()
    custom_metadata = metadata_fn()
    if custom_metadata:
        record.update(custom_metadata)

    record["stack"] = get_simplified_stack_trace(skip=2)

    # Log the record
    payload = payload_fn()
    triton_trace_log.debug("", extra={"metadata": record, "payload": payload})


def extract_python_source_info(trace_data, source, is_ir_source):
    """
    Extract Python source code information from the source object and add it to trace_data.

    Args:
        trace_data (Dict): Dictionary to store extracted information
        source: Source object (ASTSource or IRSource)
        is_ir_source (bool): Whether the source is an IR source
    """
    # Skip if source is IR or doesn't have a function attribute
    if is_ir_source or not hasattr(source, 'fn'):
        return

    import inspect
    # Get the original Python source code for the kernel
    target_fn = source.fn.fn
    python_source_file = inspect.getfile(target_fn)
    source_lines, start_line_number = inspect.getsourcelines(target_fn)
    end_line_number = start_line_number + len(source_lines)

    trace_data["python_source"] = {
        "file_path": python_source_file, "start_line": start_line_number, "end_line": end_line_number, "code":
        inspect.getsource(target_fn) if not triton_trace_no_file_content else ""
    }


def extract_file_content(trace_data, metadata_group):
    """
    Extract file content from metadata_group and add it to trace_data.

    Args:
        trace_data (Dict): Dictionary to store extracted information
        metadata_group (Dict): Dictionary mapping filenames to file paths
    """
    global triton_trace_no_file_content
    for ir_filename, file_path in metadata_group.items():
        # Add file path to trace data
        trace_data["file_path"][ir_filename] = file_path

        # Check if this is a text file we can read
        if any(ir_filename.endswith(ext) for ext in TEXT_FILE_EXTENSIONS) and not triton_trace_no_file_content:
            try:
                # Check file size before reading to avoid memory issues
                file_size = os.path.getsize(file_path)
                if file_size > MAX_FILE_SIZE:
                    trace_data["file_content"][ir_filename] = f"<file too large: {file_size} bytes>"
                    continue

                with open(file_path, 'r') as f:
                    trace_data["file_content"][ir_filename] = f.read()
            except (IOError, UnicodeDecodeError):
                # Skip if file can't be read as text
                pass


def maybe_trace_triton(metadata_path, metadata_group, src, ir_source):
    """
    Collect and trace Triton kernel compilation information for debugging and profiling.

    This function gathers metadata, IR files, and source code information about a Triton
    kernel compilation, then logs it through the tracing system if tracing is enabled.

    Args:
        metadata_path (str): Path to the JSON metadata file for the compiled kernel
        metadata_group (dict): Dictionary mapping filenames to file paths for all compilation artifacts
        src (ASTSource or IRSource): Source object containing kernel information
        ir_source (bool): Whether the source is an IR source (True) or Python source (False)

    Returns:
        dict: Dictionary containing all collected trace data, even if tracing is disabled
    """
    # Initialize a dictionary with defaultdict to avoid key errors
    trace_data = defaultdict(dict)

    # Early return if tracing is not enabled
    if not triton_trace_folder:
        return trace_data

    # Extract metadata from the JSON file if available
    if metadata_path is not None:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Copy all metadata fields to the trace data
                for key, value in metadata.items():
                    trace_data["metadata"][key] = value
        except (IOError, json.JSONDecodeError) as e:
            log.warning(f"Failed to load metadata from {metadata_path}: {e}")

    # Extract content from all IR and other files in the metadata group
    if metadata_group:
        extract_file_content(trace_data, metadata_group)

    # Extract Python source code information if available
    if src:
        extract_python_source_info(trace_data, src, ir_source)

    # Log the collected information through the tracing system
    trace_structured_triton(
        "triton.kernel",
        payload_fn=lambda: json.dumps(trace_data),
    )

    return trace_data

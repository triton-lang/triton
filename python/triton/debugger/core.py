from typing import Tuple

import dataclasses


@dataclasses.dataclass
class ExecutionContext:
    program_id: Tuple[int]
    program_size: Tuple[int]

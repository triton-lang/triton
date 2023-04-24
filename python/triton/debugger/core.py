import dataclasses


@dataclasses.dataclass
class ExecutionContext:
    program_id: tuple[int]
    program_size: tuple[int]

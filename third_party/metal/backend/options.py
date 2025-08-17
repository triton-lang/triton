class MetalOptions:
    """
    MetalOptions encapsulates configuration for Metal backend compilation and runtime.
    """
    def __init__(self, opt_level: int = 2, debug: bool = False):
        """
        Initialize MetalOptions.

        Args:
            opt_level (int): Optimization level.
            debug (bool): Enable debug mode.
        """
        self.opt_level = opt_level
        self.debug = debug
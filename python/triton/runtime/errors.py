class OutOfResources(Exception):

    def __init__(self, required, limit, name):
        self.message = f"out of resource: {name}, " f"Required: {required}, " f"Hardware limit: {limit}"
        self.message += ". Reducing block sizes or `num_stages` may help."
        self.required = required
        self.limit = limit
        self.name = name
        super().__init__(self.message)

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return (type(self), (self.required, self.limit, self.name))

class GraphRetrieveError(Exception):
    """Exception raised for errors in retrieving nodes in the graph."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DecomposeQueryError(Exception):
    """Exception raised for errors in decomposing the query."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TransformQueryError(Exception):
    """Exception raised for errors in transforming the query."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GlobalContextRetrieveError(Exception):
    """Exception raised for errors in retrieving global context."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LocalContextRetrieveError(Exception):
    """Exception raised for errors in retrieving local context."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

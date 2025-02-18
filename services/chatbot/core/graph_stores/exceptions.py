class GraphLoadError(Exception):
    """Exception raised for errors in loading the graph."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphConstructionError(Exception):
    """Exception raised for errors in constructing the graph."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)    


class GraphEntityDeduplicationError(Exception):
    """Exception raised for errors in entity deduplication."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)  


class GraphUpsertNodesError(Exception):
    """Exception raised for errors in upserting nodes in the graph."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphGetCurrentTimeError(Exception):
    """Exception raised for errors in getting the current time."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphGetTimezoneError(Exception):
    """Exception raised for errors in getting the timezone."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphProcessExtractedTripletsError(Exception):
    """Exception raised for errors in processing extracted triplets."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphAddTimestampsError(Exception):
    """Exception raised for errors in adding timestamps to nodes and relations."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphDeduplicateLlamaNodesError(Exception):
    """Exception raised for errors in deduplicating Llama nodes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphEmbedNodesError(Exception):
    """Exception raised for errors in embedding nodes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GraphUpsertEntitiesAndRelationsError(Exception):
    """Exception raised for errors in upserting entities and relations."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InitEntityRelationNodesError(Exception):
    """Exception raised for errors in initializing entity and relation nodes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

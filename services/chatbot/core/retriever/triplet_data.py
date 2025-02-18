from typing import List
from pydantic import BaseModel


class EntityInfo(BaseModel):
    """A class to store the entity information."""
    description: str = ""
    timestamp: str = ""


class RelationshipInfo(BaseModel):
    """A class to store the relationship information."""
    source_entity: str = ""
    target_entity: str = ""
    relationships: str = ""
    relationship_type: str = ""
    chunk_source_id: List[str] = []
    timestamp: str = ""


class TextChunkInfo(BaseModel):
    """A class to store the text chunk information."""
    entities: List[str] = []
    relations: List[str] = []

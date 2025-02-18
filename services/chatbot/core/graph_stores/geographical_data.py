from pydantic import BaseModel

class GeographicalData(BaseModel):
    """Geographical data model."""
    city: str = ""
    country: str = ""
    timezone: str = "UTC"

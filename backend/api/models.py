from pydantic import BaseModel  
# PURPOSE: Pydantic - validates request/response data against strict schemas, auto-generates API documentation

class OccupancyRecord(BaseModel):
    """
    Schema for a single occupancy record.
    
    Pydantic BaseModel: Automatically validates incoming data, converts types, and generates OpenAPI docs.
    This model defines the strict structure of the data that is sent to the 
    frontend dashboard and saved to the database.
    
    If data arrives without these exact fields or types, Pydantic rejects it with validation errors.
    This prevents bugs caused by malformed data.
    """
    count: int          # Pydantic type hint - The smoothed, instantaneous number of people detected in the current frame
    density: str        # Pydantic type hint - The categorical density state (e.g., "Low", "Medium", "High")
    colour: str         # Pydantic type hint - The hex colour code corresponding to the density (useful for UI styling)
    timestamp: str      # Pydantic type hint - ISO 8601 formatted timestamp of when the detection occurred
    smoothed: float     # Pydantic type hint - The Exponential Moving Average (EMA) count, used to prevent UI flickering

class ThresholdUpdate(BaseModel):
    """
    Schema for updating density classification thresholds.
    
    Used when a user dynamically adjusts the 'Low/Medium/High' boundaries 
    from an admin panel or API request without needing to restart the system.
    Pydantic ensures the incoming JSON data matches this schema.
    """
    low_max: int        # Pydantic type hint - Any count below this number is considered "Low" density
    high_min: int       # Pydantic type hint - Any count above this number is considered "High" density

class StatusResponse(BaseModel):
    """
    Schema for standard API status responses.
    
    Used for health checks and generic success/failure messages to ensure 
    the frontend knows the server is alive and functioning.
    Pydantic validates all responses conform to this schema.
    """
    status: str         # Pydantic type hint - e.g., "success" or "ok"
    message: str        # Pydantic type hint - Human-readable status message

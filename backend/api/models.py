from pydantic import BaseModel

class OccupancyRecord(BaseModel):
    """
    Schema for a single occupancy record.
    
    This model defines the strict structure of the data that is sent to the 
    frontend dashboard and saved to the database.
    """
    count: int          # The raw, instantaneous number of people detected in the current frame.
    density: str        # The categorical density state (e.g., "Low", "Medium", "High").
    colour: str         # The hex colour code corresponding to the density (useful for UI styling).
    timestamp: str      # ISO 8601 formatted timestamp of when the detection occurred.
    smoothed: float     # The Exponential Moving Average (EMA) count, used to prevent UI flickering.

class ThresholdUpdate(BaseModel):
    """
    Schema for updating density classification thresholds.
    
    This is used when a user dynamically adjusts the 'Low/Medium/High' boundaries 
    from an admin panel or API request without needing to restart the system.
    """
    low_max: int        # Any count below this number is considered "Low" density.
    high_min: int       # Any count above this number is considered "High" density.

class StatusResponse(BaseModel):
    """
    Schema for standard API status responses.
    
    Used for health checks and generic success/failure messages to ensure 
    the frontend knows the server is alive and functioning.
    """
    status: str         # e.g., "success" or "ok"
    message: str        # Human-readable status message

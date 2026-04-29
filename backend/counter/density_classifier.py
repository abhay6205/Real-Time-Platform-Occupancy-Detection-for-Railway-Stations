class DensityClassifier:
    """
    Classifies the raw passenger count into actionable density levels (Low, Medium, High).
    
    Why classify?
    A number like '42' might mean nothing to a security operator without context. 
    By translating the numerical count into semantic states ("High Density") and 
    associated colours (e.g., Red), we allow the frontend dashboard to display 
    immediate, intuitive warnings that operators can act upon instantly.
    """

    def __init__(self, low_max: int, high_min: int):
        """
        Stores the boundaries for density states.
        
        :param low_max: Any count strictly less than this is considered "Low".
        :param high_min: Any count strictly greater than or equal to this is considered "High".
                         Anything between `low_max` and `high_min` is "Medium".
        """
        self.low_max = low_max
        self.high_min = high_min

    def classify(self, count: int) -> tuple[str, str]:
        """
        Evaluates the current smoothed count against the thresholds.
        
        :param count: The current smoothed person count.
        :return: A tuple containing the text label ("Low", "Medium", "High") 
                 and its corresponding UI Hex colour code.
        """
        if count < self.low_max:
            # Green indicates safe, low occupancy.
            return "Low", "#28a745"
        elif count < self.high_min:
            # Orange indicates moderate crowding; operators should monitor.
            return "Medium", "#fd7e14"
        else:
            # Red indicates critical overcrowding; operators may need to pause escalators or dispatch staff.
            return "High", "#dc3545"

    def get_thresholds(self) -> dict:
        """
        Retrieves the active thresholds, usually requested by the frontend 
        to display current limits in the admin panel.
        """
        return {"low_max": self.low_max, "high_min": self.high_min}

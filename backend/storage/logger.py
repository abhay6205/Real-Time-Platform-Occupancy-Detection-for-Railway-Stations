import os  # PURPOSE: File system operations - creates storage directories, checks file existence
import csv  # PURPOSE: CSV file handling - writes occupancy records to human-readable CSV as Write-Ahead Log (WAL) for data recovery
import sqlite3  # PURPOSE: SQLite database - stores occupancy records in queryable relational database for efficient historical retrieval
import logging  # PURPOSE: Logging framework - records errors to console without crashing the system
from api.models import OccupancyRecord  # PURPOSE: Pydantic model - validates occupancy record schema before logging

class OccupancyLogger:
    """
    Handles dual-persistence data logging for the system.
    
    Why two databases?
    1. SQLite Database: Provides structured, easily queryable historical data. 
       Essential for the API to rapidly fetch the last N records for the frontend graph.
    2. CSV File: Acts as a Write-Ahead Log (WAL) and an ultra-portable flat file.
       If the system crashes or the SQLite database corrupts, the raw CSV ensures 
       zero data loss and allows data scientists to easily import the logs into pandas or Excel.
    """

    def __init__(self, csv_path: str, db_path: str):
        """
        Initializes the database connections and creates the files/tables if they don't exist.
        """
        self.csv_path = csv_path
        self.db_path = db_path

        # os.makedirs(): Creates storage directory if it doesn't exist
        # exist_ok=True prevents errors if directory already exists
        # Ensures both CSV and SQLite files have a valid parent directory
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # csv: Initialize the CSV file with headers if it's completely new or empty
        # Open in 'a' (append) mode to preserve existing data if file already exists
        csv_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(['timestamp', 'count', 'density', 'smoothed'])

        # sqlite3: Initialize the SQLite database and create the table schema
        # Creates occupancy table with AUTO_INCREMENT primary key for each record
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS occupancy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    count INTEGER,
                    density TEXT,
                    smoothed REAL
                )
            ''')
            conn.commit()

    def log(self, record: OccupancyRecord):
        """
        Takes an instantaneous OccupancyRecord and writes it to both the CSV and SQLite DB.
        Implements dual-persistence strategy: if one storage fails, data isn't lost.
        OccupancyRecord is already validated by Pydantic, so data is guaranteed correct schema.
        """
        try:
            # csv: Append record to the Write-Ahead Log (WAL) for crash recovery
            # CSV is human-readable and can be opened in Excel or pandas without special tools
            # Acts as first line of data persistence - written immediately for safety
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([record.timestamp, record.count, record.density, record.smoothed])

            # sqlite3: Insert the same record into structured relational database
            # sqlite3.connect(): Opens connection to database file (or creates if doesn't exist)
            # Cursor executes SQL INSERT statement with parameterized query (prevents SQL injection)
            # conn.commit(): Persists changes to disk
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO occupancy (timestamp, count, density, smoothed)
                    VALUES (?, ?, ?, ?)
                ''', (record.timestamp, record.count, record.density, record.smoothed))
                conn.commit()
        except Exception as e:
            # Non-fatal error handling: Log it to the console but keep the system alive
            # logging: Records error without crashing detection loop
            # This ensures video processing continues even if disk is full or database is corrupted
            logging.error(f"Error logging record to disk: {e}")

    def get_recent(self, n: int = 100) -> list[dict]:
        """
        Queries the SQLite database for the most recent N records.
        Used primarily if the API needs to reconstruct history after a restart.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # `sqlite3.Row` allows us to access columns by name (like a dictionary)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch records sorted descending (newest first), limited to N
                cursor.execute('''
                    SELECT * FROM occupancy
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (n,))
                rows = cursor.fetchall()
                
                # Convert the Row objects into standard Python dictionaries
                return [dict(row) for row in rows]
        except Exception as e:
            logging.error(f"Error retrieving recent records: {e}")
            return []

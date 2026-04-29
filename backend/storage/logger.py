import os
import csv
import sqlite3
import logging
from api.models import OccupancyRecord

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

        # Ensure the parent directory (e.g. 'storage/') actually exists before writing files
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize the CSV file with headers if it's completely new or empty
        csv_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not csv_exists or os.path.getsize(self.csv_path) == 0:
                writer.writerow(['timestamp', 'count', 'density', 'smoothed'])

        # Initialize the SQLite database and create the table schema
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
        This is wrapped in a try/except block so that a disk write failure doesn't 
        crash the live AI video stream.
        """
        try:
            # 1. Append to the CSV Write-Ahead log
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([record.timestamp, record.count, record.density, record.smoothed])

            # 2. Insert into the structured SQLite Database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO occupancy (timestamp, count, density, smoothed)
                    VALUES (?, ?, ?, ?)
                ''', (record.timestamp, record.count, record.density, record.smoothed))
                conn.commit()
        except Exception as e:
            # Non-fatal error handling: Log it to the console but keep the system alive
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

import sqlite3
from sqlite3 import Error
from datetime import datetime
import time

# CONSTANTS

FILE = "images.db"
IMAGES = "Messages"


class DataBase:
    """
    used to connect, write to and read from a local sqlite3 database
    """
    def __init__(self):
        """
        try to connect to file and create cursor
        """
        self.conn = None
        try:
            self.conn = sqlite3.connect(FILE)
        except Error as e:
            print(e)

        self.cursor = self.conn.cursor()
        self._create_table()

    def close(self):
        """
        close the db connection
        :return: None
        """
        self.conn.close()

    def _create_table(self):
        """
        create new database table if one doesn't exist
        :return: None
        """
        query = f"""CREATE TABLE IF NOT EXISTS {IMAGES}
                    (content_image TEXT, style_image TEXT, output_image TEXT,time Date, id INTEGER PRIMARY KEY AUTOINCREMENT)"""
        self.cursor.execute(query)
        self.conn.commit()

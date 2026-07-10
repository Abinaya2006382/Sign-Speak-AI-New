import os
import sqlite3
from datetime import datetime

# Resolve the absolute path of the database file
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'database'))
DB_PATH = os.path.join(DB_DIR, 'sign_speak.db')

def init_db():
    """Initializes the database, creating directories and tables if they don't exist."""
    # Ensure the database directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create gestures table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            spoken INTEGER DEFAULT 0
        )
    ''')
    
    # Insert default gestures individually if they don't already exist
    default_gestures = [
        ('Hello', 'Open hand, palm facing forward, fingers spread.'),
        ('Thank You', 'Hand moves from lips down and forward towards the listener.'),
        ('Yes', 'Fist nodding up and down, mimicking a head nod.'),
        ('No', 'Index and middle fingers snap down onto the thumb.'),
        ('I Love You', 'Thumb, index, and pinky fingers extended, middle and ring down.'),
        ('Thumbs Up', 'Fist with thumb pointing straight up.'),
        ('Thumbs Down', 'Fist with thumb pointing straight down.'),
        ('Bye', 'Open hand tilted, fingers slightly separated, waving pose.'),
        ('Help', 'Flat hand, palm facing up horizontally.'),
        ('Sorry', 'Closed fist with thumb pressed flat against the fingers.'),
        ('Welcome', 'Open hand with fingers relaxed and slightly curved inward.'),
        ('Good Morning', 'Flat hand with fingers closed together, pointing vertically.')
    ]
    for name, desc in default_gestures:
        cursor.execute('SELECT COUNT(*) FROM gestures WHERE name = ?', (name,))
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO gestures (name, description) VALUES (?, ?)', (name, desc))
            print(f"Seeded default gesture: {name}")
        
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")

def get_db_connection():
    """Helper function to get a SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def get_all_gestures():
    """Fetches all gestures from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM gestures ORDER BY name ASC')
    gestures = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return gestures

def add_gesture(name, description=""):
    """Adds a new gesture to the library."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO gestures (name, description) VALUES (?, ?)', (name, description))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

def delete_gesture(name):
    """Deletes a gesture from the library by its name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM gestures WHERE name = ?', (name,))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0

def get_history(limit=50):
    """Fetches the recent history of sign recognitions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT ?', (limit,))
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def add_history_entry(gesture_name, confidence, spoken=0):
    """Adds a new log to the translation history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO history (gesture_name, confidence, spoken) VALUES (?, ?, ?)',
        (gesture_name, confidence, spoken)
    )
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    return entry_id

def clear_history():
    """Deletes all items in the translation history."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history')
    conn.commit()
    conn.close()

def update_spoken_status(entry_id, spoken=1):
    """Updates whether a historical entry was read out loud."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE history SET spoken = ? WHERE id = ?', (spoken, entry_id))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Initialize DB if run directly
    init_db()

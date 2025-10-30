import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect("air_quality.db")

# Create a cursor object
cursor = conn.cursor()

# Get all table names in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in the database:")
for table in tables:
    print("-", table[0])

print("\n")

# Loop through each table and fetch its data and column names
for table in tables:
    table_name = table[0]
    print(f"--- Table: {table_name} ---")

    # Get column names
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]
    print("Columns:", columns)

    # Fetch all data
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()

    # Print data (limit output if large)
    for row in rows[:10]:  # shows only first 10 rows
        print(row)

    print("\n")

# Close the connection
conn.close()

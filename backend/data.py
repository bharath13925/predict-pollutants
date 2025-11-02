import sqlite3

# Path to your database file
db_path = "air_quality_gee.db"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("📊 Tables in database:")
for table in tables:
    print(f"  - {table[0]}")
print()

# For each table, show its schema and last 6 rows
for table in tables:
    table_name = table[0]
    print(f"\n{'='*60}")
    print(f"Table: {table_name}")
    print('='*60)
    
    # Get column information
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    
    print("\n📋 Columns:")
    for col in columns:
        col_id, name, dtype, notnull, default, pk = col
        print(f"  {col_id}. {name} ({dtype}){' [PRIMARY KEY]' if pk else ''}")
    
    # Get last 6 rows — assuming increasing rowid or primary key
    cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 6;")
    rows = cursor.fetchall()[::-1]  # reverse to show from oldest → newest
    
    print(f"\n📄 Last {len(rows)} rows:")
    print("-" * 60)
    
    # Print column headers
    col_names = [col[1] for col in columns]
    print(" | ".join(col_names))
    print("-" * 60)
    
    # Print rows
    for row in rows:
        print(" | ".join(str(val) for val in row))
    
    # Show total count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    total = cursor.fetchone()[0]
    print(f"\n📈 Total rows in {table_name}: {total}")

# Close the connection
conn.close()
print(f"\n✅ Database inspection complete!")

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# Load the CSV data
file_path = r"C:\Users\Admin\Documents\GitHub\crypto-signal-bot\test4.csv"
print(f"Loading CSV data from: {file_path}")
try:
    csv_data = pd.read_csv(file_path)
    print(f"CSV data loaded successfully with {len(csv_data)} rows.")
except Exception as e:
    print(f"Error loading CSV data: {e}")
    exit()

# Database connection parameters
db_params = {
    "dbname": "kol-tool-v1",
    "user": "postgres",
    "password": "ba8DUPGV5cIufXIef6np",
    "host": "database-2.ce6qhznpf2i4.us-east-1.rds.amazonaws.com",
    "port": "5432"
}

# Establish a connection to the database
print("Establishing database connection...")
try:
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    print("Database connection established successfully.")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# Ensure all necessary columns exist in the database
print("Ensuring database schema...")
try:
    cursor.execute("""
    DO $$ 
    BEGIN
        IF NOT EXISTS (
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'influencer' 
            AND column_name = 'niche2'
        ) THEN
            ALTER TABLE public.influencer
            ADD COLUMN "niche2" TEXT;
        END IF;
    END $$;
    """)
    conn.commit()
    print("Database schema ensured successfully.")
except Exception as e:
    conn.rollback()
    print(f"Error ensuring database schema: {e}")

# Fetch existing influencer names
print("Fetching existing influencer names...")
try:
    cursor.execute("SELECT name FROM public.influencer")
    existing_names = set(row[0] for row in cursor.fetchall())
    print(f"Fetched {len(existing_names)} existing influencer names.")
except Exception as e:
    print(f"Error fetching existing influencer names: {e}")
    conn.close()
    exit()

# Clean and validate numeric columns
numeric_columns = {
    "Followers": (-2147483648, 2147483647),
    "Individual Price": (-2147483648, 2147483647),
    "Tweet Scout Score": (-2147483648, 2147483647),
}

# Truncate values exceeding range and log changes
for column, (min_val, max_val) in numeric_columns.items():
    if column in csv_data.columns:
        csv_data[column] = pd.to_numeric(csv_data[column], errors='coerce')
        csv_data[column] = csv_data[column].clip(lower=min_val, upper=max_val)
        print(f"Ensured column '{column}' is within range {min_val} to {max_val}.")

# Convert influencer names to strings
csv_data['Influencer'] = csv_data['Influencer'].astype(str)

# Prepare queries
update_query = """
UPDATE public.influencer
SET
    "updatedAt" = NOW(),
    niche = %s,
    "niche2" = %s,
    "categoryName" = %s,
    subscribers = %s,
    geography = %s,
    platform = %s,
    price = %s,
    "credibilityScore" = %s,
    "engagementRate" = %s,
    "investorType" = %s,
    "dpLink" = %s,
    "socialMediaLink" = %s,
    "tweetScoutScore" = %s,
    deleted = FALSE
WHERE name = %s
"""

insert_query = """
INSERT INTO public.influencer (
    name, niche, "niche2", "categoryName", subscribers, geography,
    platform, price, "credibilityScore", "engagementRate", "investorType",
    "dpLink", "socialMediaLink", "tweetScoutScore", deleted, "updatedAt"
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, FALSE, NOW()
)
"""

# Process CSV data for updates and inserts
print("Processing CSV data...")
update_data = []
insert_data = []
try:
    for _, row in csv_data.iterrows():
        record = (
            row['Niche'],
            row['Niche 2'],
            row['Category'],
            row['Followers'],
            row['Geography'],
            row['Platform'],
            row['Individual Price'],
            row['Credibilty Score'],
            row['Engagement Rate'],
            row['Investor Type'],
            row['DP Links'],
            row['Link'],
            int(row['Tweet Scout Score']) if row['Tweet Scout Score'] else 0,  # Convert to float
        )

        influencer_name = row['Influencer']
        if influencer_name in existing_names:
            update_data.append(record + (influencer_name,))
        else:
            insert_data.append((influencer_name,) + record)
    
    print(f"Prepared {len(update_data)} records for update and {len(insert_data)} for insert.")
except Exception as e:
    print(f"Error processing CSV data: {e}")
    conn.close()
    exit()

# Execute updates
if update_data:
    print("Executing batch update...")
    try:
        execute_batch(cursor, update_query, update_data)
        conn.commit()
        print(f"Successfully updated {len(update_data)} records.")
    except Exception as e:
        conn.rollback()
        print(f"Error during batch update: {e}")

# Execute inserts
if insert_data:
    print("Executing batch insert...")
    try:
        execute_batch(cursor, insert_query, insert_data)
        conn.commit()
        print(f"Successfully inserted {len(insert_data)} new records.")
    except Exception as e:
        conn.rollback()
        print(f"Error during batch insert: {e}")

# Close the database connection
print("Closing database connection...")
try:
    cursor.close()
    conn.close()
    print("Database connection closed.")
except Exception as e:
    print(f"Error closing the database connection: {e}")

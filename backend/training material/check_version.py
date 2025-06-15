import pandas as pd

# Path to your original CSV
csv_path = r'/symptom2disease/backend/isic_metadata.csv'
# Path for the new, cleaned CSV
cleaned_csv_path = r'/symptom2disease/backend/isic_metadata.csv'

# Load the CSV
df = pd.read_csv(csv_path)

# Sort by isic_id (you can change to another column if you wish)
df_sorted = df.sort_values(by='isic_id')

# Remove duplicates based on isic_id, keeping the first occurrence
df_unique = df_sorted.drop_duplicates(subset=['isic_id'], keep='first')

# Save the cleaned CSV
df_unique.to_csv(cleaned_csv_path, index=False)

print(f"Sorted and deduplicated CSV saved to {cleaned_csv_path}")
print(f"Rows before: {len(df)} | Rows after: {len(df_unique)}")

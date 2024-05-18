import json
from collections import defaultdict

# Specify the absolute file path to your JSON file
file_path = '/Users/wtdoan/Downloads/via_project_17May2024_16h35m57s.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract and categorize timestamps
categorized_timestamps = defaultdict(list)
metadata = data.get("metadata", {})

for entry in metadata.values():
    timestamps = entry.get("z", [])
    labels = entry.get("av", {}).values()  # Get all labels
    for label in labels:
        for timestamp in timestamps:
            categorized_timestamps[label].append(timestamp)

# Sort and print timestamps within each category
for label, timestamps in categorized_timestamps.items():
    print(f"{label}")
    for timestamp in sorted(timestamps):
        print(f"  {timestamp}")

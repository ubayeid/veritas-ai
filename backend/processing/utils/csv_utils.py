"""
CSV processing utilities.
"""

import csv
from typing import List


def csv_to_text(csv_path: str) -> str:
    """
    Convert CSV file to text format.
    Each row is converted to a readable text format with column names.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Text representation of the CSV data
        
    Raises:
        Exception: If CSV cannot be read
        ValueError: If CSV is empty or has no columns
    """
    text_rows = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
            # Try to detect delimiter
            sample = file.read(1024)
            file.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(file, delimiter=delimiter)
            
            # Get column names
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"No columns found in CSV file: {csv_path}")
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (row 1 is header)
                # Create a text representation of the row
                row_text_parts = []
                for key, value in row.items():
                    if value and str(value).strip():  # Only include non-empty values
                        # Clean up the value
                        clean_value = str(value).strip().replace('\n', ' ').replace('\r', ' ')
                        row_text_parts.append(f"{key}: {clean_value}")
                
                if row_text_parts:
                    row_text = f"Row {row_num}: " + " | ".join(row_text_parts)
                    text_rows.append(row_text)
    
    except Exception as e:
        raise Exception(f"Error reading CSV {csv_path}: {str(e)}")
    
    if not text_rows:
        raise ValueError(f"No data rows found in CSV file: {csv_path}")
    
    return "\n".join(text_rows)

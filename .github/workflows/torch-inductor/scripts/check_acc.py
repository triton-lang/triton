import csv
import sys

file_path = sys.argv[1]
with open(file_path) as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        if row[3] != "pass":
            print(f"{row[1]} failed on device {row[0]} with batch size {row[2]}")

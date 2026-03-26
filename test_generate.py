"""Quick test: generate synthetic data."""
import sys
sys.path.insert(0, ".")
from src.data_generation.generate_synthetic import generate_all

data = generate_all(save=True)
print(f"\nTransactions: {len(data['transactions']):,} rows")
print(f"Retailers: {len(data['retailers'])} | SKUs: {len(data['skus'])}")
print("SUCCESS")

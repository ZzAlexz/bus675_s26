import pandas as pd

# Create a simple DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [85, 92, 78]
})

print("Hello from Docker!")
print(f"\nPandas version: {pd.__version__}")
print(f"\nDataFrame:\n{df}")
print(f"\nMean score: {df['score'].mean():.2f}")
import os
import numpy as np
from sklearn.datasets import fetch_openml

# === 1. Directory where your existing code expects the .npy files ===
data_dir = "datasets/MNIST/MNIST_subset"
os.makedirs(data_dir, exist_ok=True)

print(f"Saving dataset files to: {data_dir}")

# === 2. Download MNIST from OpenML (done only once) ===
print("Downloading MNIST from OpenML... This might take a minute the first time.")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

# Raw data: 70,000 samples of 784 features (28x28)
X_full = mnist.data.astype(np.float32) / 255.0     # normalize to [0,1]
digits = mnist.target.astype(np.int64)             # labels as integers

# === 3. Create wbflag array (placeholder logic) ===
# IMPORTANT:
# Replace this block with your real logic if your project requires it.
# For now, wbflag is simply an array of zeros with the same length as 'digits'.
wbflag = np.zeros_like(digits, dtype=np.int64)

# === 4. Save all arrays in .npy format with the exact filenames your code loads ===
np.save(os.path.join(data_dir, "MNIST_train_wbX.npy"), X_full)
np.save(os.path.join(data_dir, "MNIST_train_digits.npy"), digits)
np.save(os.path.join(data_dir, "MNIST_train_wbflag.npy"), wbflag)

print("\nDone! The following files were created:")
print("  - MNIST_train_wbX.npy")
print("  - MNIST_train_digits.npy")
print("  - MNIST_train_wbflag.npy")
print("\nYou can now run your project normally.")

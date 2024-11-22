import pickle
import numpy as np
import tiktoken
import os

def vectorized_multiplication_representation(num1, num2):
    """
    Returns:
    str: Formatted multiplication representation
    """
    # Validate input (5-digit numbers)
    if not (10000 <= num1 <= 99999 and 10000 <= num2 <= 99999):
        raise ValueError("Both numbers must be 5-digit numbers")
    
    # Convert number to string and extract digits
    num2_digits = np.array([int(d) for d in reversed(str(num2))])
    
    # Vectorized partial multiplications
    intermediate_calculations = num1 * num2_digits
    
    # Perform final multiplication
    result = num1 * num2
    
    # Create the formatted output string
    output = f"{num1}*{num2} = {list(intermediate_calculations)} => {result}"
    
    return output


# Data generation func
def generate_data(num_samples):
    samples = []
    for _ in range(num_samples):
        num1 = np.random.randint(1000, 10000)  # Random 5-digit numbers
        num2 = np.random.randint(1000, 10000)
        sample = vectorized_multiplication_representation(num1, num2)
        samples.append(sample)
    return samples

# Convert to batch
def preprocess_data(samples): #, output_file
    tokenized_data = []
    for sample in samples:
        tokenized_data.append(sample)
        tokenized_data.append("\n")  # Add newline token
    return "".join(tokenized_data)
    #np.array(tokenized_data, dtype=np.uint16).tofile(output_file)

data_dir = "./"
os.makedirs(data_dir, exist_ok=True)
samples = generate_data(500_000)

data = preprocess_data(samples) #, os.path.join(data_dir, "train.bin")

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
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

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Data generation func
def generate_data(num_samples):
    samples = []
    for _ in range(num_samples):
        num1 = np.random.randint(10000, 100000)  # Random 5-digit numbers
        num2 = np.random.randint(10000, 100000)
        sample = vectorized_multiplication_representation(num1, num2)
        samples.append(sample)
    return samples

# Convert to batch
def preprocess_data(samples, output_file):
    tokenized_data = []
    for sample in samples:
        tokenized = tokenizer.encode(sample)  # Tokenize using GPT-2 encoding
        tokenized_data.extend(tokenized)
        tokenized_data.append(tokenizer.encode("\n")[0])  # Add newline token
    np.array(tokenized_data, dtype=np.uint16).tofile(output_file)

data_dir = "./"
os.makedirs(data_dir, exist_ok=True)
train_samples = generate_data(100_000)
val_samples = generate_data(10_000)

preprocess_data(train_samples, os.path.join(data_dir, "train.bin"))
preprocess_data(val_samples, os.path.join(data_dir, "val.bin"))
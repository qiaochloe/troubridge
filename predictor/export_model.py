#!/usr/bin/env python3

# Export PyTorch model to TorchScript format for C++ inference.
# This script loads the trained model and tokenizer, then exports them
# in a format that can be loaded from C++ using libtorch.

import os
import sys
import torch
import torch.nn as nn
import tokenizers

model_path = "best_model.pt"
tokenizer_path = "tokenizer.json"
output_model_path = "hotness_predictor.ts"

# Add parent directory to path to import model definition
sys.path.insert(0, os.path.dirname(__file__))

# Model definition
class HotnessPredictor(nn.Module):
    def __init__(self, embedding_dim=32, hidden_size=64, num_layers=1, num_categories=3, vocab_size=2788):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size + 1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_categories)
    
    def forward(self, tokens, average_size):
        out = self.embedding(tokens)
        _, out = self.gru(out)
        out = out.squeeze(0)
        out = self.fc(torch.cat((out, average_size), dim=1))
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer file {tokenizer_path} not found")
        sys.exit(1)
    
    # Load tokenizer to get vocab size
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"Tokenizer vocab size: {vocab_size}")
    
    # Create model and load weights
    device = torch.device("cpu")  # Always use CPU for inference
    model = HotnessPredictor(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Model loaded successfully")
    
    # Create a wrapper that includes tokenization logic
    class ModelWrapper(nn.Module):
        def __init__(self, model, tokenizer):
            super().__init__()
            self.model = model
            # Store vocab as a simple dict for TorchScript compatibility
            # We'll need to handle tokenization in C++ separately
            
        def forward(self, token_ids, average_size):
            return self.model(token_ids, average_size)
    
    wrapper = ModelWrapper(model, tokenizer)
    wrapper.eval()
    
    # Test with dummy input
    dummy_tokens = torch.randint(0, vocab_size, (1, 10), dtype=torch.long)
    dummy_size = torch.tensor([[5.0]], dtype=torch.float32)
    
    with torch.no_grad():
        output = wrapper(dummy_tokens, dummy_size)
        print(f"Test output shape: {output.shape}")
        print(f"Test output: {output}")
    
    # Export to TorchScript
    print("Exporting to TorchScript...")
    traced_model = torch.jit.trace(wrapper, (dummy_tokens, dummy_size))
    traced_model.save(output_model_path)
    print(f"Model exported successfully to {output_model_path}")

if __name__ == "__main__":
    main()
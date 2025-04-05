import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar

class SequenceModel(nn.Module):
    """
    A basic sequence model that uses 8-bit Embedding, LSTM, and Linear layers with gradient checkpointing.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=6, dropout=0.5):

        super(SequenceModel, self).__init__()
        
        # 8-bit Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(torch.float32)  # Use full precision for embeddings
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers)
        
        # Fully connected layers with dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, tokens):

        embedded = self.embedding(tokens).to(torch.float32)  # Convert back to full precision for computation
        
        lstm_output, _ = torch.utils.checkpoint.checkpoint(self.lstm, embedded, use_reentrant=False)  # Explicitly set use_reentrant=False
        output = self.fc(lstm_output)
        
        return output

def train_model(model, dataloader, criterion, device, num_epochs=10, lr=1e-3, weight_decay=1e-2, max_grad_norm=1.0):
    """
    Train the SequenceModel using AdamW optimizer with gradient checkpointing and gradient clipping.
    """
    model.to(device)
    model.train()

    # Initialize AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(dataloader, desc="Training", leave=False)  # Add progress bar

        for batch in progress_bar:
            tokens, targets = batch
            tokens, targets = tokens.to(device), targets.to(device)

            # Ensure valid input tokens
            if tokens.isnan().any() or tokens.isinf().any():
                print("Invalid tokens detected. Skipping batch.")
                continue

            optimizer.zero_grad()
            outputs = model(tokens)
            
            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            if loss.isnan() or loss.isinf():
                print("Invalid loss detected. Skipping batch.")
                continue

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # Detach loss before converting to scalar
            batch_loss = loss.detach().item()
            epoch_loss += batch_loss

            # Update progress bar with batch loss
            progress_bar.set_postfix(loss=batch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")


import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceModel(nn.Module):
    """
    A basic sequence model that uses Embedding, LSTM, and Linear layers.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=6, dropout=0.5):

        super(SequenceModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
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

        embedded = self.embedding(tokens)
        
        lstm_output, _ = self.lstm(embedded)
        output = self.fc(lstm_output)
        
        return output

def train_model(model, dataloader, criterion, device, num_epochs=10, lr=1e-3, weight_decay=1e-2):
    """
    Train the SequenceModel using AdamW optimizer.
    """
    model.to(device)
    model.train()

    # Initialize AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            tokens, targets = batch
            tokens, targets = tokens.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(tokens)
            
            # Reshape outputs and targets for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")


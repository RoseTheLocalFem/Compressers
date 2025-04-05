import torch
from transformers import AutoTokenizer
from model import SequenceModel, train_model
from fineweb import load_fineweb_dataset, prepare_dataloader

def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token

    # Load dataset and prepare DataLoader
    dataset = load_fineweb_dataset(split="train", streaming=True)
    dataloader = prepare_dataloader(dataset, tokenizer, batch_size=1, max_length=8192)

    # Define model parameters
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 2048
    output_dim = vocab_size
    num_layers = 6
    dropout = 0.5

    # Initialize model
    model = SequenceModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

    # Define training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    train_model(model, dataloader, criterion, device)

if __name__ == "__main__":
    main()

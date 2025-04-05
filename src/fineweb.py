from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

def load_fineweb_dataset(name="CC-MAIN-2024-10", split="train", streaming=True):
    """Load the fineweb dataset from Hugging Face."""
    if not streaming:
        raise ValueError("Streaming mode must be enabled. Do not download the dataset.")
    return load_dataset("HuggingFaceFW/fineweb", name=name, split=split, streaming=streaming)

def extract_text_from_fineweb_entry(entry):
    """Extract only the text content from a fineweb dataset entry."""
    return entry.get('text', '')

class FineWebDataset(IterableDataset):
    """Custom Dataset for FineWeb data."""
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for entry in self.dataset:
            text = extract_text_from_fineweb_entry(entry)
            tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
            input_ids = tokens["input_ids"].squeeze(0)
            labels = input_ids.clone()  # For language modeling, input_ids are the targets
            yield input_ids, labels

def prepare_dataloader(dataset, tokenizer, batch_size=32, max_length=128):
    """Prepare a DataLoader for the FineWeb dataset."""
    fineweb_dataset = FineWebDataset(dataset, tokenizer, max_length)
    return DataLoader(fineweb_dataset, batch_size=batch_size)

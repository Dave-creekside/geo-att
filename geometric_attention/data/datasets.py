"""
Dataset classes for various NLP tasks.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class SST2Dataset(Dataset):
    """SST-2 sentiment classification dataset"""
    
    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        # Handle both full datasets and sliced dictionaries
        if isinstance(hf_dataset, dict):
            # Sliced dataset returns a dict, convert to list of dicts
            self.data = [
                {'sentence': hf_dataset['sentence'][i], 'label': hf_dataset['label'][i]}
                for i in range(len(hf_dataset['sentence']))
            ]
        else:
            self.data = hf_dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if isinstance(self.data, list):
            item = self.data[idx]
        else:
            item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item['sentence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


class MNLIDataset(Dataset):
    """MNLI natural language inference dataset"""
    
    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        # Handle both full datasets and sliced dictionaries
        if isinstance(hf_dataset, dict):
            # Sliced dataset returns a dict, convert to list of dicts
            self.data = [
                {'premise': hf_dataset['premise'][i], 
                 'hypothesis': hf_dataset['hypothesis'][i],
                 'label': hf_dataset['label'][i]}
                for i in range(len(hf_dataset['premise']))
            ]
        else:
            self.data = hf_dataset
            
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if isinstance(self.data, list):
            item = self.data[idx]
        else:
            item = self.data[idx]

        # Tokenize premise + hypothesis
        encoding = self.tokenizer(
            item['premise'],
            item['hypothesis'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


class NERDataset(Dataset):
    """Named Entity Recognition dataset"""
    
    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        tokens = item['tokens']
        ner_tags = item['ner_tags']

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_idx:
                if word_idx < len(ner_tags):
                    aligned_labels.append(ner_tags[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)  # Ignore continuation tokens
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


class LanguageModelingDataset(Dataset):
    """Language modeling dataset for causal LM"""
    
    def __init__(self, hf_dataset, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Filter out empty texts
        if isinstance(hf_dataset, list):
            self.data = hf_dataset
        else:
            self.data = [item for item in hf_dataset if len(item.get('text', '').strip()) > 0]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if isinstance(self.data[idx], dict):
            text = self.data[idx].get('text', '')
        else:
            text = str(self.data[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)

        # For language modeling: labels are input_ids shifted by 1
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = -100  # Ignore last token (no next token to predict)

        # Also mask padding tokens in labels
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def load_glue_dataset(task_name: str):
    """Load a GLUE dataset by name"""
    from datasets import load_dataset
    
    if task_name.lower() == 'sst2':
        return load_dataset("glue", "sst2")
    elif task_name.lower() == 'mnli':
        return load_dataset("glue", "mnli")
    else:
        raise ValueError(f"Unknown GLUE task: {task_name}")


def load_wikitext_dataset(version: str = "wikitext-2-raw-v1"):
    """Load WikiText dataset"""
    from datasets import load_dataset
    
    if version == "wikitext-2-raw-v1":
        return load_dataset("wikitext", "wikitext-2-raw-v1")
    elif version == "wikitext-103-raw-v1":
        return load_dataset("wikitext", "wikitext-103-raw-v1")
    else:
        raise ValueError(f"Unknown WikiText version: {version}")


def load_wikiann_dataset(language: str = "en"):
    """Load WikiANN NER dataset"""
    from datasets import load_dataset
    return load_dataset("wikiann", language)


def create_data_loader(dataset: Dataset, batch_size: int = 32, 
                       shuffle: bool = True, num_workers: int = 0):
    """Create a PyTorch DataLoader"""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

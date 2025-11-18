"""
Conversation dataset for multi-turn dialogue training.
Handles various conversation formats and proper masking.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import re


def parse_conversation(text: str, format_type: str = 'auto') -> List[Dict[str, str]]:
    """
    Parse conversation text into structured turns.
    
    Supported formats:
    - 'simple': "User: ... \\n Assistant: ..."
    - 'chatml': "<|user|>...<|assistant|>..."
    - 'alpaca': "### Instruction:...### Response:..."
    
    Args:
        text: Raw conversation text
        format_type: Format type or 'auto' to detect
        
    Returns:
        List of {'role': 'user'/'assistant', 'content': '...'}
    """
    
    turns = []
    
    # Auto-detect format
    if format_type == 'auto':
        if '<|user|>' in text or '<|assistant|>' in text:
            format_type = 'chatml'
        elif '### Instruction:' in text or '### Response:' in text:
            format_type = 'alpaca'
        else:
            format_type = 'simple'
    
    if format_type == 'simple':
        # Split on User:/Assistant: markers
        parts = re.split(r'(User:|Assistant:)', text)
        
        current_role = None
        current_content = []
        
        for part in parts:
            part = part.strip()
            if part == 'User:':
                if current_role and current_content:
                    turns.append({'role': current_role, 'content': ' '.join(current_content)})
                current_role = 'user'
                current_content = []
            elif part == 'Assistant:':
                if current_role and current_content:
                    turns.append({'role': current_role, 'content': ' '.join(current_content)})
                current_role = 'assistant'
                current_content = []
            elif part and current_role:
                current_content.append(part)
        
        # Add last turn
        if current_role and current_content:
            turns.append({'role': current_role, 'content': ' '.join(current_content)})
    
    elif format_type == 'chatml':
        # Parse ChatML format
        user_pattern = r'<\|user\|>(.*?)(?=<\|assistant\|>|$)'
        asst_pattern = r'<\|assistant\|>(.*?)(?=<\|user\|>|$)'
        
        user_matches = re.finditer(user_pattern, text, re.DOTALL)
        asst_matches = re.finditer(asst_pattern, text, re.DOTALL)
        
        all_matches = []
        for match in user_matches:
            all_matches.append(('user', match.group(1).strip(), match.start()))
        for match in asst_matches:
            all_matches.append(('assistant', match.group(1).strip(), match.start()))
        
        all_matches.sort(key=lambda x: x[2])
        turns = [{'role': role, 'content': content} for role, content, _ in all_matches]
    
    elif format_type == 'alpaca':
        # Parse Alpaca format
        instruction = re.search(r'### Instruction:(.*?)(?=### Input:|### Response:|$)', text, re.DOTALL)
        input_text = re.search(r'### Input:(.*?)(?=### Response:|$)', text, re.DOTALL)
        response = re.search(r'### Response:(.*?)$', text, re.DOTALL)
        
        if instruction:
            turns.append({'role': 'user', 'content': instruction.group(1).strip()})
        if response:
            turns.append({'role': 'assistant', 'content': response.group(1).strip()})
    
    return turns


class ConversationDataset(Dataset):
    """
    Dataset for multi-turn conversation training.
    Handles context windows, turn masking, and various formats.
    """
    
    def __init__(self, conversations: List, tokenizer, max_length: int = 512,
                 format_type: str = 'auto', mask_user_turns: bool = True):
        """
        Args:
            conversations: List of conversation strings or dicts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            format_type: Conversation format ('auto', 'simple', 'chatml', 'alpaca')
            mask_user_turns: If True, only train on assistant responses
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.mask_user_turns = mask_user_turns
        
        # Filter out empty conversations
        self.valid_conversations = []
        for conv in conversations:
            if isinstance(conv, dict):
                text = conv.get('text', '')
            else:
                text = str(conv)
            
            if text.strip():
                self.valid_conversations.append(text)
    
    def __len__(self) -> int:
        return len(self.valid_conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get conversation text
        if isinstance(self.valid_conversations[idx], dict):
            text = self.valid_conversations[idx].get('text', '')
        else:
            text = self.valid_conversations[idx]
        
        # Parse into turns
        turns = parse_conversation(text, self.format_type)
        
        # Build formatted conversation
        formatted_text = ""
        for turn in turns:
            formatted_text += f"{turn['role'].capitalize()}: {turn['content']}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        # Create labels for language modeling
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        # Mask user turns if requested
        if self.mask_user_turns:
            # Find user turn positions and mask them
            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            in_user_turn = False
            for i, token in enumerate(text_tokens):
                token_str = str(token).lower()
                
                if 'user' in token_str:
                    in_user_turn = True
                elif 'assistant' in token_str:
                    in_user_turn = False
                
                if in_user_turn:
                    labels[i] = -100
        
        # Mask padding
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'n_turns': len(turns)
        }


class MultiSessionDataset(Dataset):
    """
    Dataset that treats each conversation as a separate training session.
    Better for long multi-turn dialogues.
    """
    
    def __init__(self, conversations: List, tokenizer, max_length: int = 512,
                 format_type: str = 'auto'):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        
        # Pre-process all conversations
        self.processed = []
        for conv in conversations:
            text = conv.get('text', '') if isinstance(conv, dict) else str(conv)
            if text.strip():
                turns = parse_conversation(text, format_type)
                if turns:
                    self.processed.append(turns)
    
    def __len__(self) -> int:
        return len(self.processed)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        turns = self.processed[idx]
        
        # Build conversation with role tags
        conversation = ""
        for turn in turns:
            if turn['role'] == 'user':
                conversation += f"<|user|>{turn['content']}<|endturn|>"
            else:
                conversation += f"<|assistant|>{turn['content']}<|endturn|>"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'n_turns': len(turns)
        }


def create_conversation_splits(conversations: List, 
                               train_ratio: float = 0.8,
                               val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """
    Split conversations into train/val/test sets.
    
    Args:
        conversations: List of conversation strings/dicts
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        (train_convs, val_convs, test_convs)
    """
    import random
    
    # Shuffle
    convs = conversations.copy()
    random.shuffle(convs)
    
    n = len(convs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = convs[:n_train]
    val = convs[n_train:n_train+n_val]
    test = convs[n_train+n_val:]
    
    return train, val, test

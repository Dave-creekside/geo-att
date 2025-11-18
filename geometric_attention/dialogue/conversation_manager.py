"""
Conversation manager for tracking multi-turn dialogue state.
Handles context pruning, turn history, and session management.
"""

import torch
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


class ConversationManager:
    """
    Manages conversation state across multiple turns.
    Handles context window limits and turn history.
    """
    
    def __init__(self, max_context_length: int = 2048, max_turns: int = 20):
        """
        Args:
            max_context_length: Maximum characters in context
            max_turns: Maximum number of turns to keep in history
        """
        self.max_context_length = max_context_length
        self.max_turns = max_turns
        self.turns = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata = {}
    
    def add_turn(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a turn to the conversation"""
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.turns.append(turn)
        
        # Prune if exceeding max turns
        if len(self.turns) > self.max_turns:
            self._prune_old_turns()
    
    def _prune_old_turns(self):
        """Remove oldest turns to stay within limits"""
        # Keep most recent turns
        self.turns = self.turns[-self.max_turns:]
    
    def get_context(self, format_type: str = 'simple') -> str:
        """
        Get formatted context for model input.
        
        Args:
            format_type: How to format ('simple', 'chatml', 'alpaca')
            
        Returns:
            Formatted conversation context
        """
        if format_type == 'simple':
            context = ""
            for turn in self.turns:
                role = turn['role'].capitalize()
                context += f"{role}: {turn['content']}\n"
            return context
        
        elif format_type == 'chatml':
            context = ""
            for turn in self.turns:
                context += f"<|{turn['role']}|>{turn['content']}<|endturn|>"
            return context
        
        elif format_type == 'alpaca':
            # Simplified Alpaca format for dialogue
            context = ""
            for turn in self.turns:
                if turn['role'] == 'user':
                    context += f"### Input: {turn['content']}\n"
                else:
                    context += f"### Response: {turn['content']}\n"
            return context
        
        return ""
    
    def get_recent_turns(self, n: int = 5) -> List[Dict]:
        """Get the n most recent turns"""
        return self.turns[-n:]
    
    def clear(self):
        """Clear conversation history"""
        self.turns = []
    
    def save_conversation(self, filepath: str):
        """Save conversation to file"""
        data = {
            'session_id': self.session_id,
            'turns': self.turns,
            'metadata': self.metadata,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_conversation(self, filepath: str):
        """Load conversation from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.turns = data.get('turns', [])
        self.metadata = data.get('metadata', {})
        self.session_id = data.get('session_id', self.session_id)
    
    def get_statistics(self) -> Dict:
        """Get conversation statistics"""
        if not self.turns:
            return {}
        
        user_turns = [t for t in self.turns if t['role'] == 'user']
        asst_turns = [t for t in self.turns if t['role'] == 'assistant']
        
        total_chars = sum(len(t['content']) for t in self.turns)
        avg_user_length = sum(len(t['content']) for t in user_turns) / len(user_turns) if user_turns else 0
        avg_asst_length = sum(len(t['content']) for t in asst_turns) / len(asst_turns) if asst_turns else 0
        
        return {
            'total_turns': len(self.turns),
            'user_turns': len(user_turns),
            'assistant_turns': len(asst_turns),
            'total_characters': total_chars,
            'avg_user_length': avg_user_length,
            'avg_assistant_length': avg_asst_length
        }
    
    def export_for_training(self, output_file: str, format_type: str = 'jsonl'):
        """Export conversation in format suitable for training"""
        
        if format_type == 'jsonl':
            with open(output_file, 'w') as f:
                context = self.get_context('simple')
                f.write(json.dumps({'text': context}) + '\n')
        
        elif format_type == 'json':
            with open(output_file, 'w') as f:
                json.dump({
                    'conversation': self.turns,
                    'formatted': self.get_context('simple')
                }, f, indent=2)


class ConversationBuffer:
    """
    Manages a buffer of recent conversations for efficient context management.
    Implements sliding window with intelligent pruning.
    """
    
    def __init__(self, tokenizer, max_tokens: int = 1024):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.turns = []
    
    def add_turn(self, role: str, content: str):
        """Add turn and automatically prune if needed"""
        self.turns.append({'role': role, 'content': content})
        self._prune_to_fit()
    
    def _prune_to_fit(self):
        """Prune oldest turns to fit within token limit"""
        while len(self.turns) > 1:
            # Get current context
            context = self.get_formatted_context()
            
            # Check token count
            tokens = self.tokenizer.encode(context)
            
            if len(tokens) <= self.max_tokens:
                break
            
            # Remove oldest turn
            self.turns.pop(0)
    
    def get_formatted_context(self) -> str:
        """Get formatted context string"""
        context = ""
        for turn in self.turns:
            context += f"{turn['role'].capitalize()}: {turn['content']}\n"
        return context
    
    def get_token_count(self) -> int:
        """Get current token count"""
        context = self.get_formatted_context()
        return len(self.tokenizer.encode(context))
    
    def clear(self):
        """Clear buffer"""
        self.turns = []


def merge_conversations(conversations: List[ConversationManager]) -> ConversationManager:
    """Merge multiple conversation managers into one"""
    merged = ConversationManager()
    
    for conv in conversations:
        for turn in conv.turns:
            merged.add_turn(turn['role'], turn['content'], turn.get('metadata'))
    
    return merged

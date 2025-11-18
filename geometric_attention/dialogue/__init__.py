"""
Dialogue and conversation management for geometric attention models.
"""

from .conversation_dataset import ConversationDataset, parse_conversation
from .conversation_manager import ConversationManager
from .response_generator import ResponseGenerator

__all__ = [
    'ConversationDataset',
    'parse_conversation',
    'ConversationManager',
    'ResponseGenerator'
]

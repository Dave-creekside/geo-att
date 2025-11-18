"""
Response generator for multi-turn dialogue with advanced features.
Handles streaming, quality control, and conversation-aware generation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Callable, Dict
from .conversation_manager import ConversationManager


class ResponseGenerator:
    """
    Advanced response generator for dialogue systems.
    Supports streaming, quality filtering, and conversation coherence.
    """
    
    def __init__(self, model, tokenizer, device=None, max_model_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.max_model_length = max_model_length  # Model's trained max position length
    
    def generate_response(self, 
                         prompt: str,
                         max_length: int = 150,
                         temperature: float = 0.8,
                         top_k: int = 50,
                         top_p: float = 0.9,
                         repetition_penalty: float = 1.2,
                         min_length: int = 10,
                         stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate a single response with advanced sampling.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            min_length: Minimum response length
            stop_sequences: Sequences that end generation
            
        Returns:
            Generated response text
        """
        if stop_sequences is None:
            stop_sequences = ['User:', '\nUser:', 'Assistant:', '\n\n']
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids.clone()
        
        past_tokens = set()
        
        with torch.no_grad():
            for step in range(max_length):
                # Use sliding window if sequence exceeds model's max position
                if generated.size(1) > self.max_model_length:
                    # Only use the last max_model_length tokens
                    model_input = generated[:, -self.max_model_length:]
                else:
                    model_input = generated
                
                # Get logits
                outputs = self.model(model_input)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                next_token_logits = logits[0, -1, :].clone()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in past_tokens:
                        next_token_logits[token_id] /= repetition_penalty
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1))
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id and step >= min_length:
                    break
                
                past_tokens.add(next_token.item())
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for stop sequences
                current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                if any(seq in current_text[len(prompt):] for seq in stop_sequences):
                    break
                
                if generated.size(1) >= 512:
                    break
        
        # Decode and extract response
        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        # Clean up stop sequences
        for seq in stop_sequences:
            if seq in response:
                response = response.split(seq)[0].strip()
        
        return response
    
    def generate_streaming(self,
                          prompt: str,
                          callback: Callable[[str], None],
                          max_length: int = 150,
                          **kwargs) -> str:
        """
        Generate response with streaming (word-by-word callback).
        
        Args:
            prompt: Input prompt
            callback: Function called with each new token
            max_length: Maximum tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Complete generated response
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated = input_ids.clone()
        
        response_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Use sliding window if sequence exceeds model's max position
                if generated.size(1) > self.max_model_length:
                    model_input = generated[:, -self.max_model_length:]
                else:
                    model_input = generated
                
                outputs = self.model(model_input)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Simple sampling for streaming (faster)
                next_token_logits = logits[0, -1, :] / kwargs.get('temperature', 0.8)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                response_tokens.append(next_token.item())
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Decode and stream
                partial_response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                callback(partial_response)
                
                if generated.size(1) >= 512:
                    break
        
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    def generate_conversation_turn(self,
                                  conversation_manager: ConversationManager,
                                  user_input: str,
                                  **generation_kwargs) -> str:
        """
        Generate a response in context of ongoing conversation.
        
        Args:
            conversation_manager: ConversationManager with history
            user_input: Current user input
            **generation_kwargs: Parameters for generate_response()
            
        Returns:
            Assistant's response
        """
        # Add user turn to history
        conversation_manager.add_turn('user', user_input)
        
        # Build prompt from conversation context
        context = conversation_manager.get_context('simple')
        prompt = context + "Assistant:"
        
        # Generate response
        response = self.generate_response(prompt, **generation_kwargs)
        
        # Add assistant turn to history
        conversation_manager.add_turn('assistant', response)
        
        return response
    
    def batch_generate(self, 
                      prompts: List[str],
                      max_length: int = 100,
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in batches.
        
        Args:
            prompts: List of prompts
            max_length: Max tokens per response
            batch_size: Batch size for generation
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = []
            
            for prompt in batch_prompts:
                response = self.generate_response(prompt, max_length=max_length, **kwargs)
                batch_responses.append(response)
            
            responses.extend(batch_responses)
        
        return responses


def create_dialogue_prompt(conversation_history: List[Dict[str, str]], 
                          system_prompt: Optional[str] = None) -> str:
    """
    Create a properly formatted dialogue prompt.
    
    Args:
        conversation_history: List of turns with 'role' and 'content'
        system_prompt: Optional system message
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    if system_prompt:
        prompt += f"System: {system_prompt}\n\n"
    
    for turn in conversation_history:
        role = turn['role'].capitalize()
        content = turn['content']
        prompt += f"{role}: {content}\n"
    
    prompt += "Assistant:"
    
    return prompt


def extract_response_from_generation(full_text: str, prompt: str) -> str:
    """
    Extract just the assistant's response from generated text.
    
    Args:
        full_text: Complete generated text
        prompt: Original prompt
        
    Returns:
        Cleaned assistant response
    """
    # Remove prompt
    response = full_text[len(prompt):].strip()
    
    # Remove any trailing user prompts
    if "User:" in response:
        response = response.split("User:")[0].strip()
    
    # Remove newlines at start/end
    response = response.strip()
    
    return response

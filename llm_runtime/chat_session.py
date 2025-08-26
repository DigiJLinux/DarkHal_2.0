"""
Chat Session Management for KV Caching

This module provides chat session management with persistent KV cache
for efficient multi-turn conversations across different model formats.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import threading


@dataclass
class ChatSessionConfig:
    """Configuration for chat sessions"""
    max_context_length: int = 4096
    cache_warmup: bool = True  # Pre-fill system prompt
    streaming: bool = True
    

class ChatSession:
    """
    Manages persistent KV cache state for multi-turn conversations.
    
    Supports different model formats:
    - GGUF models: Rely on llama-cpp-python's built-in caching
    - Transformers models: Manual KV cache management with past_key_values
    """
    
    def __init__(self, session_id: str, config: ChatSessionConfig = None):
        self.session_id = session_id
        self.config = config or ChatSessionConfig()
        self.lock = threading.RLock()
        
        # KV cache state (transformers models)
        self.past_key_values: Optional[Tuple] = None
        self.cached_input_ids: Optional[List[int]] = None
        self.context_length: int = 0
        
        # Conversation history
        self.messages: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
        
        # State tracking
        self.is_prefilled: bool = False
        self.last_prompt_hash: Optional[str] = None
        self.last_prompt: Optional[str] = None
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        with self.lock:
            if role == "system" and not self.messages:
                self.system_prompt = content
            self.messages.append({"role": role, "content": content})
    
    def clear_messages(self) -> None:
        """Clear conversation history but preserve system prompt"""
        with self.lock:
            if self.system_prompt:
                self.messages = [{"role": "system", "content": self.system_prompt}]
            else:
                self.messages = []
            self.invalidate_cache()
    
    def invalidate_cache(self) -> None:
        """Invalidate KV cache - forces re-prefill on next generation"""
        with self.lock:
            self.past_key_values = None
            self.cached_input_ids = None
            self.context_length = 0
            self.is_prefilled = False
            self.last_prompt_hash = None
            self.last_prompt = None
    
    def should_invalidate(self, new_prompt: str, tokenizer=None) -> bool:
        """Check if cache should be invalidated based on prompt changes with proper token-level validation"""
        import hashlib
        
        current_hash = hashlib.md5(new_prompt.encode()).hexdigest()
        
        # Always invalidate if no cache exists
        if self.past_key_values is None:
            return True
            
        # Always invalidate if we don't have a tokenizer for proper validation
        if tokenizer is None:
            return True
            
        # Always invalidate if no previous prompt exists
        if not hasattr(self, 'last_prompt') or self.last_prompt is None:
            return True
            
        # Check for exact token-level prefix match
        if self._has_valid_token_prefix(new_prompt, self.last_prompt, tokenizer):
            # Additional safety checks for turn boundaries
            if self._violates_turn_boundaries(new_prompt, self.last_prompt):
                print("[KV_CACHE] Invalidating cache: turn boundary violation detected")
                return True
            return False
        
        # Invalidate if no valid prefix match
        print("[KV_CACHE] Invalidating cache: no valid token prefix match")
        return True
    
    def _has_valid_token_prefix(self, new_prompt: str, old_prompt: str, tokenizer) -> bool:
        """Check if new prompt has exact token-level prefix match with cached prompt"""
        try:
            # Tokenize both prompts
            old_tokens = tokenizer.encode(old_prompt, add_special_tokens=False)
            new_tokens = tokenizer.encode(new_prompt, add_special_tokens=False)
            
            # New prompt must be longer than or equal to old
            if len(new_tokens) < len(old_tokens):
                return False
                
            # Check exact token-by-token match for the prefix
            for i, (old_token, new_token) in enumerate(zip(old_tokens, new_tokens)):
                if old_token != new_token:
                    print(f"[KV_CACHE] Token mismatch at position {i}: old={old_token}, new={new_token}")
                    return False
            
            return True
        except Exception as e:
            print(f"[KV_CACHE] Error in token prefix validation: {e}")
            return False
    
    def _violates_turn_boundaries(self, new_prompt: str, old_prompt: str) -> bool:
        """Check if the prompt violates turn boundaries (conversation integrity)"""
        # Look for end-of-turn markers that indicate conversation corruption
        eot_markers = ["<|eot_id|>", "<|end_of_turn|>", "</s>", "<|endoftext|>"]
        
        # If the old prompt ended with an EOT marker, we should start fresh
        for marker in eot_markers:
            if old_prompt.rstrip().endswith(marker):
                # Check if new prompt is a proper continuation (should start with User: or similar)
                continuation_part = new_prompt[len(old_prompt):].lstrip()
                if not (continuation_part.startswith("User:") or continuation_part.startswith("Human:") or continuation_part.startswith("\nUser:") or continuation_part.startswith("\nHuman:")):
                    return True
        
        # Check for conversation format corruption (duplicate roles, malformed structure)
        if self._has_conversation_corruption(new_prompt):
            return True
            
        return False
    
    def _has_conversation_corruption(self, prompt: str) -> bool:
        """Detect conversation format corruption that indicates cache should be invalidated"""
        # Look for signs of corrupted conversation format
        corruption_patterns = [
            "Assistant: Assistant:",  # Duplicate assistant labels
            "User: User:",  # Duplicate user labels  
            "Assistant: User",  # Role confusion
            "User: Assistant:",  # Role confusion
            "of of of",  # Repetitive token generation (sign of corruption)
            "151 of 151",  # Specific corruption pattern we observed
        ]
        
        for pattern in corruption_patterns:
            if pattern in prompt:
                print(f"[KV_CACHE] Detected conversation corruption: '{pattern}'")
                return True
                
        return False
    
    def update_cache(self, past_key_values: Tuple, input_ids: List[int], prompt: str) -> None:
        """Update the KV cache state with turn boundary validation"""
        import hashlib
        
        with self.lock:
            # Validate that we're ending on a complete turn boundary
            if not self._is_complete_turn(prompt):
                print("[KV_CACHE] Warning: Caching incomplete turn - this may cause issues")
            
            self.past_key_values = past_key_values
            self.cached_input_ids = input_ids
            self.context_length = len(input_ids)
            self.is_prefilled = True
            self.last_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            self.last_prompt = prompt  # Store the actual prompt for comparison
    
    def _is_complete_turn(self, prompt: str) -> bool:
        """Check if the prompt ends on a complete turn boundary"""
        # Look for proper turn endings
        prompt_trimmed = prompt.rstrip()
        
        # Should end with Assistant response, not mid-generation
        valid_endings = [
            "</s>",
            "<|eot_id|>", 
            "<|end_of_turn|>",
            "<|endoftext|>",
        ]
        
        # Or should end with a complete sentence/response
        if any(prompt_trimmed.endswith(ending) for ending in valid_endings):
            return True
            
        # For non-marked conversations, check if it looks like a complete response
        # (ends with punctuation and doesn't look cut off)
        if prompt_trimmed.endswith(('.', '!', '?', ':', ';')):
            return True
            
        # If it ends with "Assistant:" it's ready for generation
        if prompt_trimmed.endswith("Assistant:"):
            return True
        
        return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        with self.lock:
            return {
                "session_id": self.session_id,
                "has_cache": self.past_key_values is not None,
                "context_length": self.context_length,
                "is_prefilled": self.is_prefilled,
                "message_count": len(self.messages),
                "max_context": self.config.max_context_length
            }


class ChatSessionManager:
    """Manages multiple chat sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.lock = threading.RLock()
        self.default_session_id = "default"
    
    def get_session(self, session_id: str = None, config: ChatSessionConfig = None) -> ChatSession:
        """Get or create a chat session"""
        if session_id is None:
            session_id = self.default_session_id
            
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = ChatSession(session_id, config)
            return self.sessions[session_id]
    
    def clear_session(self, session_id: str = None) -> None:
        """Clear a specific session"""
        if session_id is None:
            session_id = self.default_session_id
            
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].invalidate_cache()
                self.sessions[session_id].clear_messages()
    
    def remove_session(self, session_id: str) -> None:
        """Remove a session completely"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all session IDs"""
        with self.lock:
            return list(self.sessions.keys())


# Global session manager instance
_session_manager = ChatSessionManager()


def get_session_manager() -> ChatSessionManager:
    """Get the global chat session manager"""
    return _session_manager
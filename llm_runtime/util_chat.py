from typing import Any, Dict, List, Optional

def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """Apply chat template to messages using tokenizer or fallback to ChatML format"""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    # Fallback: simple ChatML-ish format
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
        else:
            parts.append(f"<|user|>\n{content}\n")
    
    if add_generation_prompt:
        parts.append("<|assistant|>\n")
    
    return "".join(parts)
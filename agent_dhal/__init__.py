#!/usr/bin/env python3
"""
AgentDhal - Complete AI Agent Framework for DarkHal 2.0

A comprehensive agent framework providing:
- Multi-agent conversation capabilities
- Agent orchestration and team management
- Tool integration and function calling
- Model context management
- Memory and state management
- Customizable agent behaviors

Legal Attribution:
This software is based on Microsoft AutoGen (https://github.com/microsoft/autogen)
Licensed under MIT License. AgentDhal is a derivative work with
modifications and extensions for the DarkHal project.

Copyright (c) 2025 DarkHal Project
"""

__version__ = "1.0.0"
__author__ = "DarkHal Project (based on Microsoft AutoGen)"

# Import core AgentDhal components
from .agentdhal_core import (
    Agent,
    AgentId,
    AgentRuntime,
    SingleThreadedAgentRuntime,
    RoutedAgent,
    MessageContext,
    DefaultTopicId,
    message_handler,
    default_subscription,
    BaseAgent,
    AgentType,
    TopicId,
    Subscription
)

# Import Dhal - our primary AI agent
from .hal import Dhal, DhalConfig, create_dhal

# Import other AgentDhal components (available but not primary focus)
try:
    from .agentdhal_agentchat import (
        AssistantAgent,
        UserProxyAgent,
        ChatAgent,
        Team
    )
except ImportError:
    # Graceful fallback if agentchat modules have issues
    AssistantAgent = None
    UserProxyAgent = None
    ChatAgent = None
    Team = None

__all__ = [
    # Core framework
    "Agent",
    "AgentId", 
    "AgentRuntime",
    "SingleThreadedAgentRuntime",
    "RoutedAgent",
    "MessageContext",
    "DefaultTopicId",
    "message_handler",
    "default_subscription",
    "BaseAgent",
    "AgentType",
    "TopicId",
    "Subscription",
    
    # Primary Hal Agent
    "Hal",
    "HalConfig",
    "create_hal",
    
    # Additional Agent Components (if available)
    "AssistantAgent",
    "UserProxyAgent", 
    "ChatAgent",
    "Team"
]
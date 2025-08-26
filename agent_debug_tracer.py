#!/usr/bin/env python3
"""
Agent Debug and Trace System for DarkAgent
Custom logging and monitoring system that tracks agent lifecycle without using sys.settrace()
"""

import time
import threading
import queue
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

class AgentDebugTracer:
    """Custom debug and trace system for DarkAgent monitoring."""
    
    def __init__(self, log_file: str = "agent_debug.log"):
        self.log_file = Path(log_file)
        self.trace_queue = queue.Queue()
        self.is_running = False
        self.logger_thread = None
        self.start_time = time.time()
        
        # Agent state tracking
        self.agent_states = {}
        self.event_history = []
        self.performance_metrics = {
            "total_messages": 0,
            "successful_responses": 0,
            "errors": 0,
            "average_response_time": 0.0,
            "response_times": []
        }
        
        # Initialize log file
        self._init_log_file()
        
    def _init_log_file(self):
        """Initialize the log file with header."""
        try:
            with open(self.log_file, 'w') as f:
                f.write(f"=== DarkAgent Debug Trace Started: {datetime.now()} ===\n")
                f.write(f"Application Launch Time: {self.start_time}\n")
                f.write("=" * 60 + "\n\n")
        except Exception as e:
            print(f"[AGENT_DEBUG] Failed to initialize log file: {e}")
    
    def start_monitoring(self):
        """Start the debug monitoring system."""
        if not self.is_running:
            self.is_running = True
            self.logger_thread = threading.Thread(target=self._logger_worker, daemon=True)
            self.logger_thread.start()
            self.trace("SYSTEM", "Debug monitoring started")
    
    def stop_monitoring(self):
        """Stop the debug monitoring system."""
        if self.is_running:
            self.trace("SYSTEM", "Debug monitoring stopping")
            self.is_running = False
            if self.logger_thread and self.logger_thread.is_alive():
                self.logger_thread.join(timeout=1.0)
    
    def trace(self, category: str, message: str, data: Dict[str, Any] = None):
        """Add a trace entry."""
        if not self.is_running:
            return
            
        timestamp = time.time()
        elapsed = timestamp - self.start_time
        
        entry = {
            "timestamp": timestamp,
            "elapsed": elapsed,
            "datetime": datetime.now().isoformat(),
            "category": category,
            "message": message,
            "data": data or {},
            "thread": threading.current_thread().name
        }
        
        try:
            self.trace_queue.put_nowait(entry)
            self.event_history.append(entry)
            
            # Keep history manageable
            if len(self.event_history) > 1000:
                self.event_history = self.event_history[-500:]
                
        except queue.Full:
            pass  # Drop trace if queue is full
    
    def _logger_worker(self):
        """Background thread that writes trace entries to file."""
        while self.is_running:
            try:
                entry = self.trace_queue.get(timeout=1.0)
                self._write_trace_entry(entry)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AGENT_DEBUG] Logger error: {e}")
    
    def _write_trace_entry(self, entry: Dict[str, Any]):
        """Write a trace entry to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                formatted_time = f"{entry['elapsed']:8.3f}s"
                thread_info = f"[{entry['thread']}]" if entry['thread'] != 'MainThread' else ""
                
                f.write(f"{formatted_time} [{entry['category']}]{thread_info} {entry['message']}")
                
                if entry['data']:
                    f.write(f" | Data: {json.dumps(entry['data'], indent=None)}")
                
                f.write("\n")
                f.flush()
                
        except Exception as e:
            print(f"[AGENT_DEBUG] Write error: {e}")
    
    # Agent-specific monitoring methods
    def agent_startup(self, agent_name: str, config: Dict[str, Any] = None):
        """Track agent startup."""
        self.agent_states[agent_name] = {
            "status": "starting",
            "start_time": time.time(),
            "config": config or {}
        }
        self.trace("AGENT_STARTUP", f"Agent {agent_name} starting", {"config": config})
    
    def agent_ready(self, agent_name: str):
        """Track agent ready state."""
        if agent_name in self.agent_states:
            self.agent_states[agent_name]["status"] = "ready"
            startup_time = time.time() - self.agent_states[agent_name]["start_time"]
            self.trace("AGENT_READY", f"Agent {agent_name} ready", {"startup_time": startup_time})
    
    def agent_message_start(self, agent_name: str, message: str, message_id: str = None):
        """Track start of message processing."""
        self.performance_metrics["total_messages"] += 1
        self.trace("AGENT_MESSAGE_START", f"Agent {agent_name} processing message", {
            "message_id": message_id,
            "message_preview": message[:100] + "..." if len(message) > 100 else message
        })
        return time.time()  # Return start time for response time calculation
    
    def agent_message_end(self, agent_name: str, message_id: str, start_time: float, success: bool = True, error: str = None):
        """Track end of message processing."""
        response_time = time.time() - start_time
        self.performance_metrics["response_times"].append(response_time)
        
        if success:
            self.performance_metrics["successful_responses"] += 1
            self.trace("AGENT_MESSAGE_SUCCESS", f"Agent {agent_name} completed message", {
                "message_id": message_id,
                "response_time": response_time
            })
        else:
            self.performance_metrics["errors"] += 1
            self.trace("AGENT_MESSAGE_ERROR", f"Agent {agent_name} message failed", {
                "message_id": message_id,
                "response_time": response_time,
                "error": error
            })
        
        # Update average response time
        if self.performance_metrics["response_times"]:
            self.performance_metrics["average_response_time"] = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
    
    def agent_shutdown(self, agent_name: str):
        """Track agent shutdown."""
        if agent_name in self.agent_states:
            self.agent_states[agent_name]["status"] = "shutdown"
            uptime = time.time() - self.agent_states[agent_name]["start_time"]
            self.trace("AGENT_SHUTDOWN", f"Agent {agent_name} shutting down", {"uptime": uptime})
    
    def agent_error(self, agent_name: str, error: str, context: Dict[str, Any] = None):
        """Track agent errors."""
        self.performance_metrics["errors"] += 1
        self.trace("AGENT_ERROR", f"Agent {agent_name} error: {error}", context)
    
    def ui_event(self, event_type: str, details: Dict[str, Any] = None):
        """Track UI events related to agent."""
        self.trace("UI_EVENT", event_type, details)
    
    def model_event(self, event_type: str, model_info: Dict[str, Any] = None):
        """Track model loading/unloading events."""
        self.trace("MODEL_EVENT", event_type, model_info)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            "uptime": time.time() - self.start_time,
            "agent_states": self.agent_states,
            "performance": self.performance_metrics,
            "recent_events": self.event_history[-10:] if self.event_history else []
        }
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.get_performance_summary()
        print("\n=== DarkAgent Debug Summary ===")
        print(f"Uptime: {summary['uptime']:.2f}s")
        print(f"Total Messages: {summary['performance']['total_messages']}")
        print(f"Successful Responses: {summary['performance']['successful_responses']}")
        print(f"Errors: {summary['performance']['errors']}")
        print(f"Average Response Time: {summary['performance']['average_response_time']:.3f}s")
        print(f"Active Agents: {len([a for a in summary['agent_states'].values() if a['status'] == 'ready'])}")
        print("=" * 31)


# Global tracer instance
_global_tracer: Optional[AgentDebugTracer] = None

def get_tracer() -> AgentDebugTracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentDebugTracer()
        _global_tracer.start_monitoring()
    return _global_tracer

def trace(category: str, message: str, data: Dict[str, Any] = None):
    """Convenience function for tracing."""
    get_tracer().trace(category, message, data)

def shutdown_tracer():
    """Shutdown the global tracer."""
    global _global_tracer
    if _global_tracer:
        _global_tracer.stop_monitoring()
        _global_tracer = None
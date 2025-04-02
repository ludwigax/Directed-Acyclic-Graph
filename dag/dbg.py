import time
from abc import ABC
from typing import Dict, Any

IS_DEBUGGING = False

def get_debug_state():
    return IS_DEBUGGING

def set_debug_state(state: bool):
    global IS_DEBUGGING
    IS_DEBUGGING = state

class Debug(ABC):
    def __init__(self):
        self.execution_time = 0.0
        self.call_count = 0
        self.last_call_time = 0.0
    
    def _start_timer(self):
        self.last_call_time = time.time()
    
    def _stop_timer(self):
        elapsed = time.time() - self.last_call_time
        self.execution_time += elapsed
        self.call_count += 1
        return elapsed
    
    def get_stats(self):
        avg_time = self.execution_time / max(1, self.call_count)
        return {
            "execution_time": self.execution_time,
            "call_count": self.call_count,
            "avg_time": avg_time
        }
    
    def reset_stats(self):
        self.execution_time = 0.0
        self.call_count = 0
        self.last_call_time = 0.0


class DebuggingContext:
    def __init__(self, enable: bool):
        self.enable = enable
        self.previous_state = None
        
    def __enter__(self):
        global IS_DEBUGGING
        self.previous_state = IS_DEBUGGING
        IS_DEBUGGING = self.enable
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global IS_DEBUGGING
        IS_DEBUGGING = self.previous_state
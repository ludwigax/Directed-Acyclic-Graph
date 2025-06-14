"""
Launcher for interactive DAG visualization
"""
import sys
import os
from typing import Optional
import webbrowser
import time
import threading

from .flask_app import create_app
from ..node import Module


def launch_visualizer(module: Module, 
                     host: str = 'localhost', 
                     port: int = 5000,
                     auto_open: bool = True,
                     debug: bool = False) -> None:
    """
    Launch the interactive DAG visualizer
    
    Args:
        module: The module to visualize
        host: Host address to bind the Flask app
        port: Port to bind the Flask app
        auto_open: Whether to automatically open browser
        debug: Whether to run Flask in debug mode
    """
    
    # Create the Flask app
    app = create_app(module)
    
    # Enable detailed error reporting
    app.app.config['PROPAGATE_EXCEPTIONS'] = True
    if debug:
        app.app.config['DEBUG'] = True
    
    # Print startup information
    print(f"Starting DAG Interactive Visualizer...")
    print(f"Module: {module.name} ({module.__class__.__name__})")
    print(f"Server: http://{host}:{port}")
    print(f"Debug mode: {'ON' if debug else 'OFF'}")
    print(f"Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Auto-open browser if requested
    if auto_open:
        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f'http://{host}:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\nShutting down DAG visualizer...")
    except Exception as e:
        print(f"Error starting server: {e}")


def visualize_module(module: Module, **kwargs) -> None:
    """
    Convenience function to visualize a module
    
    Args:
        module: The module to visualize
        **kwargs: Additional arguments passed to launch_visualizer
    """
    launch_visualizer(module, **kwargs)


if __name__ == "__main__":
    print("DAG Interactive Visualizer")
    print("This script should be used by importing visualize_module or launch_visualizer")
    print("Example usage:")
    print("  from dag.inspect_utils.launcher import visualize_module")
    print("  visualize_module(my_module)") 
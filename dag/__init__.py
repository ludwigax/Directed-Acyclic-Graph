# import sys
# from .node import (
#     Module, InspectModule, FunctionModule, ModuleGroup, Constant, NullNode, VirtualNode,
#     connect, clear_cache, returns_keys, reset_module_stats, get_module_stats
# )
# from .dbg import DebuggingContext, set_debug_state, get_debug_state
# from .inspect_utils import print_module, visualize_module

# def ensure_utf8_encoding():
#     try:
#         if sys.stdout.encoding != 'utf-8':
#             sys.stdout.reconfigure(encoding='utf-8')
#         if sys.stderr.encoding != 'utf-8':
#             sys.stderr.reconfigure(encoding='utf-8')
#     except Exception as e:
#         import io
#         sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#         sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
#     print("Encoding set to UTF-8")

# ensure_utf8_encoding()
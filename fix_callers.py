#!/usr/bin/env python3
import re
import sys

# Read the file
with open('tools/clang/tools/dxc-fuzzer/MiniHLSLInterpreter.cpp', 'r') as f:
    content = f.read()

# Pattern to find removeThreadFromNestedBlocks calls
pattern = r'(tg\.removeThreadFromNestedBlocks\([^;]+)(\);)'

# Counter for replacements
replacements = 0

def replace_callback(match):
    global replacements
    call = match.group(1)
    
    # Check if it already has 4 parameters (the new signature)
    if call.count(',') >= 3:
        return match.group(0)  # Already updated
    
    replacements += 1
    
    # Find context to determine what to pass
    # Get line number and surrounding context
    start = max(0, match.start() - 500)
    context = content[start:match.start()]
    
    # Check if we're in a loop break handler
    if any(phrase in context for phrase in [
        "handleBreakException", 
        "ControlFlowBreak from ForStmt",
        "ControlFlowBreak from WhileStmt",
        "ControlFlowBreak from DoStmt",
        "break within ForStmt",
        "break within WhileStmt",
        "break within DoStmt"
    ]):
        # This is a break from a loop - pass 'this'
        return call + ", this);"
    else:
        # Not a loop break - pass nullptr
        return call + ", nullptr);"

# Apply replacements
new_content = re.sub(pattern, replace_callback, content)

# Write back
with open('tools/clang/tools/dxc-fuzzer/MiniHLSLInterpreter.cpp', 'w') as f:
    f.write(new_content)

print(f"Updated {replacements} calls to removeThreadFromNestedBlocks")
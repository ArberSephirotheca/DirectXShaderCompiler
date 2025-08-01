#!/usr/bin/env python3
"""
Script to run all HLSL example files through the minihlsl-standalone interpreter
and capture the output to individual text files.
"""

import os
import subprocess
import glob
import re
from pathlib import Path

# Paths
BUILD_DIR = "/home/t-zheychen/dxc_workspace/DirectXShaderCompiler/build-fuzzer"
EXAMPLES_DIR = "/home/t-zheychen/dxc_workspace/DirectXShaderCompiler/tools/clang/tools/dxc-fuzzer/examples"
EXECUTABLE = os.path.join(BUILD_DIR, "bin", "minihlsl-standalone")
OUTPUT_DIR = "/home/t-zheychen/dxc_workspace/DirectXShaderCompiler/tools/clang/tools/dxc-fuzzer/test_outputs"

def parse_expected_results(hlsl_file):
    """Parse expected results from comments in HLSL file."""
    expected = {}
    try:
        with open(hlsl_file, 'r') as f:
            content = f.read()
        
        # Look for comments like "// totalSum = 54"
        total_sum_match = re.search(r'//\s*totalSum\s*=\s*(\d+)', content)
        if total_sum_match:
            expected['totalSum'] = int(total_sum_match.group(1))
            
        # Could add more patterns here for other expected values
        # result_match = re.search(r'//\s*result\s*=\s*(\d+)', content)
        # if result_match:
        #     expected['result'] = int(result_match.group(1))
            
    except Exception as e:
        print(f"Warning: Could not parse expected results from {hlsl_file}: {e}")
    
    return expected

def verify_results(output_text, expected_results):
    """Verify actual results match expected results."""
    verification_results = []
    
    for var_name, expected_value in expected_results.items():
        if var_name == 'totalSum':
            # Look for totalSum values in the output
            # Pattern: "totalSum = 54" in the variable values section
            total_sum_matches = re.findall(r'totalSum\s*=\s*(\d+)', output_text)
            
            if total_sum_matches:
                # Check if all totalSum values match the expected value
                actual_values = [int(match) for match in total_sum_matches]
                all_match = all(val == expected_value for val in actual_values)
                
                if all_match:
                    verification_results.append(f"✅ {var_name}: {expected_value} (found {len(actual_values)} instances)")
                else:
                    unique_values = set(actual_values)
                    verification_results.append(f"❌ {var_name}: expected {expected_value}, got {unique_values}")
            else:
                verification_results.append(f"⚠️  {var_name}: expected {expected_value}, but not found in output")
    
    return verification_results

def run_example(hlsl_file):
    """Run a single HLSL example file and capture output."""
    filename = Path(hlsl_file).stem
    output_file = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    
    # Parse expected results from the HLSL file
    expected_results = parse_expected_results(hlsl_file)
    
    # Command to run
    cmd = [EXECUTABLE, "-o", "1", hlsl_file]
    
    print(f"Running {filename}...")
    if expected_results:
        print(f"Expected: {expected_results}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Combine stdout and stderr
        output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        output += f"=== RETURN CODE ===\n{result.returncode}\n\n"
        
        if result.stdout:
            output += f"=== STDOUT ===\n{result.stdout}\n\n"
        
        if result.stderr:
            output += f"=== STDERR ===\n{result.stderr}\n\n"
        
        # Verify results if we have expected values
        verification_results = []
        verification_passed = True
        if expected_results and result.returncode == 0:
            verification_results = verify_results(output, expected_results)
            verification_passed = all('✅' in vr for vr in verification_results)
            
            # Add verification results to output
            if verification_results:
                output += f"=== VERIFICATION ===\n"
                for vr in verification_results:
                    output += f"{vr}\n"
                output += "\n"
        
        # Write to output file
        with open(output_file, 'w') as f:
            f.write(output)
        
        # Print summary
        if result.returncode == 0:
            if expected_results:
                if verification_passed:
                    status = "SUCCESS ✅"
                else:
                    status = "SUCCESS (verification failed) ⚠️"
            else:
                status = "SUCCESS (no verification)"
        else:
            status = "FAILED"
            
        print(f"  -> {status} (exit code: {result.returncode})")
        if verification_results:
            for vr in verification_results:
                print(f"    {vr}")
        print(f"  -> Output saved to: {output_file}")
        
        return result.returncode == 0 and verification_passed
        
    except subprocess.TimeoutExpired:
        error_output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        error_output += "=== ERROR ===\nTIMEOUT: Command took longer than 30 seconds\n"
        
        with open(output_file, 'w') as f:
            f.write(error_output)
        
        print(f"  -> TIMEOUT")
        print(f"  -> Error saved to: {output_file}")
        return False
        
    except Exception as e:
        error_output = f"=== COMMAND ===\n{' '.join(cmd)}\n\n"
        error_output += f"=== ERROR ===\n{str(e)}\n"
        
        with open(output_file, 'w') as f:
            f.write(error_output)
        
        print(f"  -> ERROR: {e}")
        print(f"  -> Error saved to: {output_file}")
        return False

def main():
    """Main function to run all examples."""
    print("Running all HLSL examples...")
    print(f"Examples directory: {EXAMPLES_DIR}")
    print(f"Executable: {EXECUTABLE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if executable exists
    if not os.path.exists(EXECUTABLE):
        print(f"ERROR: Executable not found: {EXECUTABLE}")
        return 1
    
    # Find all HLSL files
    hlsl_pattern = os.path.join(EXAMPLES_DIR, "*.hlsl")
    hlsl_files = sorted(glob.glob(hlsl_pattern))
    
    if not hlsl_files:
        print(f"ERROR: No HLSL files found in {EXAMPLES_DIR}")
        return 1
    
    print(f"Found {len(hlsl_files)} HLSL files:")
    for f in hlsl_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Run each example
    successful = 0
    failed = 0
    results = []  # Store results for summary table
    
    for hlsl_file in hlsl_files:
        filename = Path(hlsl_file).stem
        expected_results = parse_expected_results(hlsl_file)
        success = run_example(hlsl_file)
        
        # Store result for summary table
        result_entry = {
            'filename': filename,
            'success': success,
            'expected': expected_results.get('totalSum', 'N/A') if expected_results else 'N/A'
        }
        results.append(result_entry)
        
        if success:
            successful += 1
        else:
            failed += 1
        print()  # Empty line for readability
    
    # Summary
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total examples: {len(hlsl_files)}")
    print(f"  Successful (with verification): {successful}")
    print(f"  Failed: {failed}")
    print()
    
    # Generate summary table
    print("DETAILED RESULTS:")
    print("=" * 80)
    print(f"{'Test Case':<35} {'Expected':<12} {'Status':<10}")
    print("-" * 80)
    
    for result in results:
        filename = result['filename']
        expected = str(result['expected'])
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        
        # Truncate long filenames
        display_name = filename if len(filename) <= 34 else filename[:31] + "..."
        
        print(f"{display_name:<35} {expected:<12} {status:<10}")
    
    print("-" * 80)
    print(f"{'TOTAL':<35} {'':<12} {successful}/{len(hlsl_files)} PASS")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All examples ran successfully with correct results!")
        return 0
    else:
        print(f"⚠️  {failed} examples failed")
        return 1

if __name__ == "__main__":
    exit(main())
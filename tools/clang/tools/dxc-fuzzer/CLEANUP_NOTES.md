# Cleanup Notes for HLSL Fuzzer Integration

## Summary
The incremental fuzzer functionality has been successfully integrated into minihlsl-fuzzer.
The minihlsl-fuzzer now supports two modes:
1. Traditional seed corpus mode (default)
2. Random program generation mode (enabled with FUZZ_GENERATE_RANDOM=1 environment variable)

## Files That Can Be Removed

### 1. fuzz_hlsl_incremental.cpp
- **Reason**: The separate incremental fuzzer entry point is now redundant
- **Replacement**: Use `FUZZ_GENERATE_RANDOM=1 ./bin/minihlsl-fuzzer` instead

### 2. HLSLSemanticMutator.cpp/h
- **Reason**: These files implemented mutations but weren't using traces properly
- **Replacement**: Mutations are correctly implemented in MiniHLSLInterpreterFuzzer.cpp

### 3. MiniHLSLInterpreterFuzzerMutations.cpp
- **Reason**: This was created to copy mutations from MiniHLSLInterpreterFuzzer.cpp but is redundant
- **Note**: Still used by fuzz-hlsl-file target, but not needed for minihlsl-fuzzer

## Files That Must Be Kept

### For Random Generation Support
- HLSLProgramGenerator.cpp/h
- HLSLProgramGeneratorUtils.cpp
- HLSLParticipantPatterns.cpp/h
- HLSLMutationTracker.cpp/h (needed by HLSLProgramGenerator)

### Core Fuzzer Files
- MiniHLSLInterpreter.cpp/h
- MiniHLSLInterpreterFuzzer.cpp/h
- MiniHLSLInterpreterTraceCapture.cpp/h
- MiniHLSLValidator.cpp/h

## Usage

### Traditional Mode (with seed corpus)
```bash
./bin/minihlsl-fuzzer seeds -runs=1000
```

### Random Generation Mode
```bash
FUZZ_GENERATE_RANDOM=1 ./bin/minihlsl-fuzzer -runs=1000
```

## Build Configuration
The CMakeLists.txt has been updated to include all necessary files for random generation
in the minihlsl-fuzzer target. The fuzz-hlsl-incremental target has been removed.
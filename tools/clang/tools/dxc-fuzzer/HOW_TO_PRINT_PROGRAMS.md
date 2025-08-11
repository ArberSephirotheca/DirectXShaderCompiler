# How to Print Original and Mutated Programs

The minihlsl-fuzzer already prints both original and mutated programs by default when running mutations.

## Usage

### 1. Run with existing seed corpus
```bash
./bin/minihlsl-fuzzer seeds -runs=10
```

### 2. Run with random program generation

When using random generation mode, you need to provide sufficient input data (at least 16 bytes):

```bash
# Create a random input file
dd if=/dev/urandom of=test_input bs=1024 count=1 2>/dev/null

# Run fuzzer with random generation
FUZZ_GENERATE_RANDOM=1 ./bin/minihlsl-fuzzer test_input -runs=1
```

Note: Without an explicit input file, libFuzzer starts with very small inputs that may not trigger random generation.

## Output Format

When the fuzzer runs, it will output:

```
=== Testing Mutant 1 (Strategy: LanePermutation) ===

--- Original Program ---
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint value = tid.x;
  uint result = WaveActiveSum(value);
}

--- Mutant Program ---
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint value = tid.x;
  if(true) {
    uint _perm_lane = laneIndex() ^ 1;
    uint _perm_val_0 = WaveReadLaneAt(value, _perm_lane);
    result = WaveActiveSum(_perm_val_0);
  }
}
--- End Mutant Program ---
```

## What You'll See

1. **Original Program**: The input program before any mutations
2. **Mutant Program**: The program after applying semantic-preserving mutations
3. **Strategy Name**: Which mutation strategy was applied (e.g., LanePermutation, WaveParticipantTracking)

## Mutation Strategies

The fuzzer applies various semantic-preserving mutations:

- **LanePermutation**: Permutes lane data before wave operations (for associative operations)
- **WaveParticipantTracking**: Adds tracking to verify correct participant counts
- **PrecomputeWaveResults**: Precomputes wave operation results (when trace data available)
- **ExplicitLaneDivergence**: Makes implicit divergence explicit with conditionals

## Notes

- Programs are printed for every mutant tested
- The output can be verbose with many mutations
- Use `-runs=1` to see just one mutation cycle
- The fuzzer validates that mutations preserve semantics by comparing execution traces
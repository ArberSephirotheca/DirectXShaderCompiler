// Control flow with wave operations
// Tests deterministic branching

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    if (laneId < 16) {
        result = 1;  // First half of wave
        if (laneId == 0) {
            // Special case for lane 0
            result += 10;  // Just an arbitrary operation
        }
    } else {
        result = 2;  // Second half of wave
    }
    
    // Wave sum should be: (16 * 1) + (16 * 2) = 48
    uint totalSum = WaveActiveSum(result);
}
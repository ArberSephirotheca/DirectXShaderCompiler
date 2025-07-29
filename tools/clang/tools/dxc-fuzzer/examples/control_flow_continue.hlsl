// Control flow with wave operations
// Tests deterministic branching

[numthreads(5, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    for (int i = 0; i < 4; ++i) {
        if (laneId == i || laneId == 1) {
            continue;
        }
        result += WaveActiveSum(1);
    }
    
    // Expected per-lane results:
    // Lane 0: skip at i=0, result=3
    // Lane 1: skip all iterations, result=0
    // Lane 2: skip at i=2, result=3
    // Lane 3: skip at i=3, result=3
    // Lane 4: never skip, result=4
    // WaveActiveSum: 0+0+2+3+4 = 9
    
    uint totalSum = WaveActiveSum(result);
}
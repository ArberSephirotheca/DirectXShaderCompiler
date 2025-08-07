// Control flow with wave operations
// Tests deterministic branching
// totalSum = 21
[numthreads(4, 1, 1)]
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
    // Lane 0: skip at i=0, result= 3 + 2 + 2 = 7
    // Lane 1: skip all iterations, result=0
    // Lane 2: skip at i=2, result= 3 + 2 + 2 = 7
    // Lane 3: skip at i=3, result= 3 + 2 + 2 = 7
    // WaveActiveSum: 7+0+7+7 = 21 
    
    uint totalSum = WaveActiveSum(result);
}
// Control flow with wave operations
// Tests deterministic branching

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    int i = 0;
    // Deterministic branching - all lanes take predictable paths
    while (i < 4) {
        if (laneId == i || laneId == 1) {
            i += 1;
            continue;
        }
        result += WaveActiveSum(1);
        i += 1;
    }
    
    // Expected per-lane results:
    // Lane 0: skip at i=0, result= 3 + 2 + 2 = 7
    // Lane 1: breaks at i=0, result=0  
    // Lane 2: breaks at i=2, result= 2 + 2 + 3 = 7
    // Lane 3: breaks at i=3, result= 2 + 2 + 3 = 7
    // WaveActiveSum: 7 + 0 + 7 + 7 = 21
    
    uint totalSum = WaveActiveSum(result);
}
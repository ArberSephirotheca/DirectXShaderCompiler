// Control flow with wave operations
// Tests deterministic branching without buffer operations

[numthreads(5, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    for (int i = 0; i < 4; ++i) {
        if (laneId == i || laneId == 1) {
            break;
        }
        result += 1;
    }
    
    // Expected values per lane:
    // Lane 0: breaks at i=0, result=0
    // Lane 1: breaks at i=0, result=0  
    // Lane 2: breaks at i=2, result=2
    // Lane 3: breaks at i=3, result=3
    // Lane 4: never breaks, result=4
    // Total sum: 0+0+2+3+4 = 9
    
    uint totalSum = WaveActiveSum(result);
    
    // Store in a simple variable instead of buffer
    uint finalValue = totalSum;
}
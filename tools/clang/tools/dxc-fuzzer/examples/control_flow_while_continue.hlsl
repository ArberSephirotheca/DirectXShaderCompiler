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
            i++;
            continue;
        }
        result += WaveActiveSum(1);
        i++;
    }
    
    // Expected per-lane results:
    // Lane 0: breaks at i=0, result=0
    // Lane 1: breaks at i=0, result=0  
    // Lane 2: breaks at i=2, result= 2 + 2 = 4
    // Lane 3: breaks at i=3, result= 2 + 2 + 1 = 5
    // WaveActiveSum: 0+0+2+4+5 = 9
    
    uint totalSum = WaveActiveSum(result);
}
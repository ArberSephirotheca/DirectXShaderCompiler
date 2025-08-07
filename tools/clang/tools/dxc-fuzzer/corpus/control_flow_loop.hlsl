// Control flow with wave operations
// Tests deterministic branching
// totalSum = 14
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    for (int i = 0; i < laneId; ++i) {
        result += WaveActiveSum(1);
    }
    
    uint totalSum = WaveActiveSum(result);
}
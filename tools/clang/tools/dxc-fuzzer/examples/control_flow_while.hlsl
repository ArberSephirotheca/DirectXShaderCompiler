// Control flow with wave operations
// Tests deterministic branching
// totalSum = 14
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    int i = 0;
    // Deterministic branching - all lanes take predictable paths
    while (i < laneId) {
        result += WaveActiveSum(1);
        i++;
    }
    
    uint totalSum = WaveActiveSum(result);
}
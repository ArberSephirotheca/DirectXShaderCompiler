// Control flow with wave operations
// Tests deterministic branching

[numthreads(5, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    int i = 0;
    // Deterministic branching - all lanes take predictable paths
    while (i < 4) {
        result += WaveActiveSum(1);
        i++;
    }
    
    // Wave sum should be: (16 * 1) + (16 * 2) = 48
    uint totalSum = WaveActiveSum(result);
}
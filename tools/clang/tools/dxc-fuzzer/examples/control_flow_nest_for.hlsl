// Control flow with wave operations
// Tests deterministic branching

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    // (4 * 2 + 4 * 4)  * 4 = 96
    for (int i = 0; i < 2; ++i) {
        result += WaveActiveSum(1);
        for (int j = 0; j < 2; ++j) {
            result += WaveActiveSum(1);
        }
    }
    uint totalSum = WaveActiveSum(result);
}
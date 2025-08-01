// Control flow with wave operations
// Tests deterministic branching

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    if (laneId < 2) {
        for (int i = 0; i < 2; ++i) {
            result += WaveActiveSum(1);
        }
        result += WaveActiveSum(1);
    } else {
        result += WaveActiveSum(5);
    }
    // 10 + 10 + 4 + 4 = 28
    uint totalSum = WaveActiveSum(result);
}
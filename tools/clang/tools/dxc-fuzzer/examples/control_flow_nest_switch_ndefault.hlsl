// Control flow with wave operations
// Tests deterministic branching

[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    int i = 0;
    // Deterministic branching - all lanes take predictable paths
    switch (laneId) {
        case 0:
        result += WaveActiveSum(0);
        case 1:
        result += WaveActiveSum(1);
        case 2:
        result += WaveActiveSum(2);
        case 3:
        result += WaveActiveSum(3);
    }
    
    // Wave sum should be: (16 * 1) + (16 * 2) = 48
    uint totalSum = WaveActiveSum(result);
}
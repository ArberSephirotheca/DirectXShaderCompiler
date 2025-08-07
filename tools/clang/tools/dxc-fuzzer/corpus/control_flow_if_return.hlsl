// Control flow with wave operations
// Tests deterministic branching
// totalSum = 11
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Deterministic branching - all lanes take predictable paths
    if (laneId < 2) {
        // result += WaveActiveSum(2);
        if (laneId == 0){
            return;
            result += WaveActiveSum(5);
        } else {
            result += WaveActiveSum(1);
        }
        result += WaveActiveSum(2);
    } else {
        result += WaveActiveSum(2);  // Second half of wave
    }
    
    // Wave sum should be: 3 + 4 + 4 = 11
    uint totalSum = WaveActiveSum(result);
}
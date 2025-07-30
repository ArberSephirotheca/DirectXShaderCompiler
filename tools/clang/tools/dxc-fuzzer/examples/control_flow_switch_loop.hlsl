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
          for (int b = 0; b < 3; b++) {
              result += WaveActiveSum(b);
          }
        case 1:
        result += WaveActiveSum(1);
        case 2:
        result += WaveActiveSum(2);
        case 3:
        result += WaveActiveSum(3);
        default:
        result += WaveActiveSum(10);

    }
    
    uint totalSum = WaveActiveSum(result);
}
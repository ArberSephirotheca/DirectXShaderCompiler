// Simple test for break without wave operations
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    for (int i = 0; i < 4; ++i) {
        if (i == 2) {
            break;
        }
        result += 1;
    }
    
    // Expected: all lanes should have result = 2
    uint totalSum = WaveActiveSum(result);
}
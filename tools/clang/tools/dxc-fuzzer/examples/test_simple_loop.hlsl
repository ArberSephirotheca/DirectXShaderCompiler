// Very simple loop test without break

[numthreads(2, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = 0;
    
    // Simple loop - no break
    for (int i = 0; i < 3; ++i) {
        result = result + 1;
    }
    
    // result should be 3 for both lanes
    uint sum = WaveActiveSum(result);
    // sum should be 6 (3 + 3)
}
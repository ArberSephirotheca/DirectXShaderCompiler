// Test barrier synchronization
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = laneId;
    
    // All threads should wait here until everyone arrives
    GroupMemoryBarrierWithGroupSync();
    
    // After barrier, all threads should see consistent state
    result += 10;
    
    uint totalSum = WaveActiveSum(result);
    // Expected: (0+10) + (1+10) + (2+10) + (3+10) = 46
}
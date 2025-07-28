// Test barrier deadlock detection - some threads exit early
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint result = laneId;
    
    // Only some threads reach the barrier - this should cause a deadlock error
    if (laneId < 2) {
        return; // Threads 0 and 1 exit early without hitting barrier
    }
    
    // Only threads 2 and 3 reach this barrier
    GroupMemoryBarrierWithGroupSync(); // This should trigger deadlock detection
    
    result += 10;
    uint totalSum = WaveActiveSum(result);
}
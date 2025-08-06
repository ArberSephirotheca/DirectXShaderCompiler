// Nested control flow with wave operations
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint laneId = tid.x % 32;
    
    int value = 0;
    
    if (laneId < 16) {
        if (laneId < 8) {
            value = 1;
        } else {
            value = 2;
        }
        
        // Wave op in nested branch
        value = WaveActiveSum(value);
    } else {
        value = 3;
        
        // Different wave op in other branch
        value = WaveActiveMax(value);
    }
    
    // Barrier to synchronize
    GroupMemoryBarrierWithGroupSync();
    
    // Final wave op after reconvergence
    int final = WaveActiveSum(value);
    
    RWBuffer<int> output : register(u0);
    output[tid.x] = final;
}
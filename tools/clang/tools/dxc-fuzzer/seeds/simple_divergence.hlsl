// Simple divergence test for fuzzer
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint laneId = tid.x % 32;
    uint waveId = tid.x / 32;
    
    int x = 0;
    
    // Simple divergence
    if (laneId < 16) {
        x = 1;
    } else {
        x = 2;
    }
    
    // Wave operation
    int sum = WaveActiveSum(x);
    
    // Store result
    RWBuffer<int> output : register(u0);
    output[tid.x] = sum;
}
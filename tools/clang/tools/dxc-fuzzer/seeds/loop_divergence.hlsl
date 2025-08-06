// Loop with divergent iteration count
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint laneId = tid.x % 32;
    
    int sum = 0;
    
    // Different lanes iterate different number of times
    for (int i = 0; i < laneId % 4 + 1; i++) {
        sum += i;
    }
    
    // Wave operation inside divergent control flow
    int waveSum = WaveActiveSum(sum);
    
    RWBuffer<int> output : register(u0);
    output[tid.x] = waveSum;
}
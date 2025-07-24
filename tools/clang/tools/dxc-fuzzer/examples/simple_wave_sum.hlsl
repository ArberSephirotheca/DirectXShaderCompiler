// Simple wave reduction example
// Each thread contributes its lane ID to a wave sum

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
    uint value = laneId * laneId;  // Each lane contributes square of its ID
    uint waveSum = WaveActiveSum(value);
    
    // All lanes in the wave should get the same sum
    // For 32 lanes: sum = 0² + 1² + 2² + ... + 31² = 10416
}
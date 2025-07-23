// Invalid MiniHLSL Test Case: Multi-prefix operations
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid11() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Multi-prefix operations are order-dependent
    // float multiPrefix = WaveMultiPrefixSum(float(lane), lane < 16);  // ERROR
}
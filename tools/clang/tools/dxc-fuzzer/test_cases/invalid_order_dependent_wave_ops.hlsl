// Invalid MiniHLSL Test Case: Order-dependent wave operations
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid1() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Prefix operations are order-dependent
    float prefix = WavePrefixSum(1.0f);
    float prefixProduct = WavePrefixProduct(float(lane));
}
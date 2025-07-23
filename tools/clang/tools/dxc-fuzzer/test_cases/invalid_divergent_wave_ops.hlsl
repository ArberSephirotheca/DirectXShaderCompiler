// Invalid MiniHLSL Test Case: Wave operations in divergent control flow
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid2() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Wave-divergent condition before wave operation
    if (lane % 2 == 0) {  // Not all lanes participate
        float sum = WaveActiveSum(1.0f);  // ERROR: Incomplete participation
    }
}
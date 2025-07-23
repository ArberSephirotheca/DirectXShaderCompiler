// Invalid MiniHLSL Test Case: Non-uniform loop with wave operations
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid3() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Loop count varies per lane
    for (int i = 0; i < lane; ++i) {  // Wave-divergent loop
        float value = WaveActiveSum(float(i));  // ERROR: Wave op in divergent loop
    }
}
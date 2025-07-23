// Invalid MiniHLSL Test Case: While loop with wave-dependent condition
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid6() {
    uint lane = WaveGetLaneIndex();
    uint counter = 0;
    
    // ERROR: Wave-dependent loop condition
    while (counter < lane) {  // Different termination per lane
        counter++;
        float sum = WaveActiveSum(float(counter));  // ERROR: Wave op in divergent loop
    }
}
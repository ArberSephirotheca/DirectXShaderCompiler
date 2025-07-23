// Invalid MiniHLSL Test Case: Switch with divergent conditions
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid5() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Switch on lane-dependent value
    switch (lane & 3) {  // Wave-divergent switch
        case 0: {
            float sum = WaveActiveSum(1.0f);  // ERROR: Not all lanes participate
            break;
        }
        case 1: {
            float product = WaveActiveProduct(2.0f);  // ERROR: Different lanes
            break;
        }
        default:
            break;
    }
}
// Valid MiniHLSL Test Case: Uniform branching with wave queries
// This should pass validation - order-independent program

[numthreads(32, 1, 1)]
void main3() {
    uint lane = WaveGetLaneIndex();
    float value = float(lane);
    
    // All uniform conditions
    if (WaveActiveAllEqual(42)) {
        if (WaveActiveAnyTrue(true)) {
            if (WaveGetLaneCount() >= 32) {
                float result = WaveActiveSum(value);
            }
        }
    }
}
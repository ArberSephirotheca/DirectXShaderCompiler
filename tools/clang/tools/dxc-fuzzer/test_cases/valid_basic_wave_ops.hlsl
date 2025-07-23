// Valid MiniHLSL Test Case: Basic uniform wave operations
// This should pass validation - order-independent program

[numthreads(32, 1, 1)]
void main1() {
    uint lane = WaveGetLaneIndex();
    
    // Order-independent arithmetic
    float value = float(lane * lane);
    
    // Commutative reduction
    float sum = WaveActiveSum(value);
    
    // Uniform condition
    if (WaveGetLaneCount() == 32) {
        float average = sum / 32.0f;
        bool isAboveAverage = value > average;
        uint count = WaveActiveCountBits(isAboveAverage);
    }
}
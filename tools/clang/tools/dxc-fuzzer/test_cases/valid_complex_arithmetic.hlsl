// Valid MiniHLSL Test Case: Complex order-independent arithmetic
// This should pass validation - order-independent program

[numthreads(32, 1, 1)]
void main4() {
    uint lane = WaveGetLaneIndex();
    
    // Commutative operations preserve order-independence
    float a = float(lane) + float(lane * 2);
    float b = a * 3.0f + 1.0f;
    
    // All lanes participate in wave operations
    float sum = WaveActiveSum(a);
    float product = WaveActiveProduct(b);
    
    // Order-independent condition
    bool evenLane = (lane & 1) == 0;
    uint evenCount = WaveActiveCountBits(evenLane);
}
// Valid MiniHLSL Test Case: Multiple associative operations
// This should pass validation - order-independent program

[numthreads(64, 1, 1)]
void main2() {
    uint idx = WaveGetLaneIndex();
    
    // Associative operations (order-independent)
    uint product = WaveActiveProduct(idx + 1);
    uint maxVal = WaveActiveMax(idx);
    uint minVal = WaveActiveMin(idx);
    
    // Additional arithmetic operations (associative)
    float floatSum = WaveActiveSum(float(idx));
    bool evenLane = (idx & 1) == 0;
    uint evenCount = WaveActiveCountBits(evenLane);
}
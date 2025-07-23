// Valid MiniHLSL Test Case: Deterministic expressions only
// This should pass validation - order-independent program

[numthreads(32, 1, 1)]
void main5() {
    uint lane = WaveGetLaneIndex();
    
    // Deterministic arithmetic
    float x = float(lane * lane + lane);
    float y = x / (x + 1.0f);
    
    // Deterministic comparisons
    bool condition = x > y;
    
    // Order-independent wave operations
    if (WaveIsFirstLane()) {
        // Only first lane executes, but no wave ops inside
        float temp = x + y;
    }
    
    // All lanes participate
    float maxX = WaveActiveMax(x);
}
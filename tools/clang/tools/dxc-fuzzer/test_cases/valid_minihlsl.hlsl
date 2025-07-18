// Valid MiniHLSL Test Cases - Order-Independent Programs

// Test Case 1: Basic uniform wave operations
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

// Test Case 2: Multiple associative operations
[numthreads(64, 1, 1)]
void main2() {
    uint idx = WaveGetLaneIndex();
    
    // Associative operations (order-independent)
    uint product = WaveActiveProduct(idx + 1);
    uint maxVal = WaveActiveMax(idx);
    uint minVal = WaveActiveMin(idx);
    
    // Bitwise operations (associative)
    uint andResult = WaveActiveAnd(idx);
    uint orResult = WaveActiveOr(idx);
    uint xorResult = WaveActiveXor(idx);
}

// Test Case 3: Uniform branching with wave queries
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

// Test Case 4: Complex order-independent arithmetic
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

// Test Case 5: Deterministic expressions only
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
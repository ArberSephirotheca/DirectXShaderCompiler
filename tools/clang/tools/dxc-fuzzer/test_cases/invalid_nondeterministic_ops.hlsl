// Invalid MiniHLSL Test Case: Non-deterministic operations
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid10() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Non-deterministic function calls (if they existed)
    // float randomValue = random();  // Would be error
    
    // ERROR: Undefined behavior expressions
    float undefined = 1.0f / 0.0f;  // ERROR: Division by zero - non-deterministic
}
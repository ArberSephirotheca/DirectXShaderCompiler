// Invalid MiniHLSL Test Case: Memory racing patterns
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid8() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Potential data race (simplified - would need actual memory)
    // This represents unsynchronized memory writes that could race
    if (lane % 2 == 0) {
        // Hypothetical write to shared memory location
        // Would cause data race between lanes
    }
}
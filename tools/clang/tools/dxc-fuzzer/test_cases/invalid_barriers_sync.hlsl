// Invalid MiniHLSL Test Case: Barriers and synchronization
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid12() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Barriers are forbidden in MiniHLSL
    // AllMemoryBarrier();  // ERROR: Barriers forbidden
    // GroupMemoryBarrier();  // ERROR: Barriers forbidden
}
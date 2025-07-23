// Invalid MiniHLSL Test Case: Forbidden language constructs
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid9() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Forbidden 'for' loop (should use specific loop types)
    for (int i = 0; i < 10; ++i) {  // ERROR: 'for' is forbidden in MiniHLSL
        float value = float(i);
    }
    
    // ERROR: 'break' and 'continue' are forbidden
    // break;  // Would be error if uncommented
}
// Invalid MiniHLSL Test Cases - Order-Dependent Programs (Should be rejected)

// INVALID Test Case 1: Order-dependent wave operations
[numthreads(32, 1, 1)]
void invalid1() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Prefix operations are order-dependent
    float prefix = WavePrefixSum(1.0f);
    float prefixProduct = WavePrefixProduct(float(lane));
}

// INVALID Test Case 2: Wave operations in divergent control flow
[numthreads(32, 1, 1)]
void invalid2() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Wave-divergent condition before wave operation
    if (lane % 2 == 0) {  // Not all lanes participate
        float sum = WaveActiveSum(1.0f);  // ERROR: Incomplete participation
    }
}

// INVALID Test Case 3: Non-uniform loop with wave operations
[numthreads(32, 1, 1)]
void invalid3() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Loop count varies per lane
    for (int i = 0; i < lane; ++i) {  // Wave-divergent loop
        float value = WaveActiveSum(float(i));  // ERROR: Wave op in divergent loop
    }
}

// INVALID Test Case 4: Lane-dependent read operations
[numthreads(32, 1, 1)]
void invalid4() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Order-dependent lane reads
    float firstLaneValue = WaveReadFirstLane(float(lane));
    float specificLaneValue = WaveReadLaneAt(float(lane), 5);
}

// INVALID Test Case 5: Switch with divergent conditions
[numthreads(32, 1, 1)]
void invalid5() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Switch on lane-dependent value
    switch (lane & 3) {  // Wave-divergent switch
        case 0: {
            float sum = WaveActiveSum(1.0f);  // ERROR: Not all lanes participate
            break;
        }
        case 1: {
            float product = WaveActiveProduct(2.0f);  // ERROR: Different lanes
            break;
        }
        default:
            break;
    }
}

// INVALID Test Case 6: While loop with wave-dependent condition
[numthreads(32, 1, 1)]
void invalid6() {
    uint lane = WaveGetLaneIndex();
    uint counter = 0;
    
    // ERROR: Wave-dependent loop condition
    while (counter < lane) {  // Different termination per lane
        counter++;
        float sum = WaveActiveSum(float(counter));  // ERROR: Wave op in divergent loop
    }
}

// INVALID Test Case 7: Ballot operations (order-dependent usage)
[numthreads(32, 1, 1)]
void invalid7() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Ballot operations can be order-dependent
    uint4 ballot = WaveBallot(lane < 16);  // ERROR: Order-dependent ballot
}

// INVALID Test Case 8: Memory racing patterns
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

// INVALID Test Case 9: Forbidden language constructs
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

// INVALID Test Case 10: Non-deterministic operations
[numthreads(32, 1, 1)]
void invalid10() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Non-deterministic function calls (if they existed)
    // float randomValue = random();  // Would be error
    
    // ERROR: Undefined behavior expressions
    float undefined = 1.0f / 0.0f;  // ERROR: Division by zero - non-deterministic
}

// INVALID Test Case 11: Multi-prefix operations
[numthreads(32, 1, 1)]
void invalid11() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Multi-prefix operations are order-dependent
    // float multiPrefix = WaveMultiPrefixSum(float(lane), lane < 16);  // ERROR
}

// INVALID Test Case 12: Barriers and synchronization
[numthreads(32, 1, 1)]
void invalid12() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Barriers are forbidden in MiniHLSL
    // AllMemoryBarrier();  // ERROR: Barriers forbidden
    // GroupMemoryBarrier();  // ERROR: Barriers forbidden
}
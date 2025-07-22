// Valid test cases for Dynamic Block Execution Graph (DBEG) construction
// These tests should pass validation

RWBuffer<float> buffer : register(u0);

//===----------------------------------------------------------------------===//
// Test 1: Simple if-else with disjoint memory writes
//===----------------------------------------------------------------------===//
[numthreads(32, 1, 1)]
void dbeg_test_simple_if_else(uint tid : SV_DispatchThreadID) {
    if (tid < 16) {
        buffer[tid] = 1.0f;        // Threads 0-15 write to [0-15]
    } else {
        buffer[tid + 16] = 2.0f;   // Threads 16-31 write to [32-47]
    }
    buffer[tid + 48] = 3.0f;       // All threads write to [48-79]
}

//===----------------------------------------------------------------------===//
// Test 2: Early return with disjoint writes
//===----------------------------------------------------------------------===//
[numthreads(16, 1, 1)]
void dbeg_test_early_return(uint tid : SV_DispatchThreadID) {
    if (tid < 8) {
        buffer[tid] = 1.0f;
        return; // Threads 0-7 exit
    }
    
    // Only threads 8-15 reach here
    buffer[tid + 8] = 2.0f;   // Write to [16-23]
    buffer[tid + 24] = 3.0f;  // Write to [32-39]
}

//===----------------------------------------------------------------------===//
// Test 3: Nested control flow with returns
//===----------------------------------------------------------------------===//
[numthreads(16, 1, 1)]
void dbeg_test_nested_returns(uint tid : SV_DispatchThreadID) {
    if (tid < 12) {
        if (tid < 8) {
            buffer[tid] = 1.0f;
            if (tid < 4) {
                return; // Threads 0-3 exit
            }
            buffer[tid + 16] = 2.0f;  // Threads 4-7
        } else {
            buffer[tid + 32] = 3.0f;  // Threads 8-11
        }
        buffer[tid + 48] = 4.0f;      // Threads 4-11
    } else {
        buffer[tid + 64] = 5.0f;      // Threads 12-15
    }
    buffer[tid + 80] = 6.0f;          // Threads 4-15
}

//===----------------------------------------------------------------------===//
// Test 4: Deterministic loop with disjoint writes
//===----------------------------------------------------------------------===//
[numthreads(8, 1, 1)]
void dbeg_test_loop_disjoint(uint tid : SV_DispatchThreadID) {
    for (int i = 0; i < 4; i++) {
        if (tid < 4) {
            buffer[tid * 4 + i] = float(i);        // Threads 0-3: [0-15]
        } else {
            buffer[(tid + 8) * 4 + i] = float(i);  // Threads 4-7: [48-63]
        }
    }
    buffer[tid + 96] = 99.0f;  // All threads: [96-103]
}

//===----------------------------------------------------------------------===//
// Test 5: Complex but valid interleaved pattern
//===----------------------------------------------------------------------===//
[numthreads(16, 1, 1)]
void dbeg_test_interleaved(uint tid : SV_DispatchThreadID) {
    if (tid < 8) {
        buffer[tid * 2] = 1.0f;      // Even indices: 0,2,4,6,8,10,12,14
    } else {
        buffer[tid * 2 + 1] = 2.0f;  // Odd indices: 17,19,21,23,25,27,29,31
    }
    buffer[tid + 128] = 3.0f;        // [128-143], no overlap
}
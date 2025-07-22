// Invalid test case: Cross-path dependency violation
// This should be detected by DBEG analysis

RWBuffer<float> data : register(u0);

//===----------------------------------------------------------------------===//
// Cross-dynamic-block dependency: Write in one branch, read in another
//===----------------------------------------------------------------------===//
[numthreads(32, 1, 1)]
void invalid_cross_path_dependency(uint tid : SV_DispatchThreadID) {
    // This violates order independence because:
    // - Thread 0 writes to data[0] in the first branch
    // - Thread 16 reads from data[0] in the second branch
    // - These are in different dynamic blocks with no synchronization
    
    if (tid < 16) {
        // Dynamic Block 1: Threads 0-15
        data[tid] = 1.0f;  // Thread 0 writes to data[0]
    } else {
        // Dynamic Block 2: Threads 16-31
        float val = data[tid - 16];  // Thread 16 reads from data[0]
        data[tid] = val + 2.0f;
    }
    
    // The read-after-write dependency crosses dynamic blocks
    // This creates a race condition that violates MiniHLSL constraints
}
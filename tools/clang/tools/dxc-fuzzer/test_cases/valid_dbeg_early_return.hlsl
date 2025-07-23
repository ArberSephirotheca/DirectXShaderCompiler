// Valid DBEG Test Case: Early return with disjoint writes
// This should pass validation - order-independent program

RWBuffer<float> buffer : register(u0);

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
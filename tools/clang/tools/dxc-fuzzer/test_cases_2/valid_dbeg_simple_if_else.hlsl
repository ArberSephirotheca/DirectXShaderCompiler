// Valid DBEG Test Case: Simple if-else with disjoint memory writes
// This should pass validation - order-independent program

RWBuffer<float> buffer : register(u0);

[numthreads(32, 1, 1)]
void dbeg_test_simple_if_else(uint tid : SV_DispatchThreadID) {
    if (tid < 16) {
        buffer[tid] = 1.0f;        // Threads 0-15 write to [0-15]
    } else {
        buffer[tid + 16] = 2.0f;   // Threads 16-31 write to [32-47]
    }
    buffer[tid + 48] = 3.0f;       // All threads write to [48-79]
}
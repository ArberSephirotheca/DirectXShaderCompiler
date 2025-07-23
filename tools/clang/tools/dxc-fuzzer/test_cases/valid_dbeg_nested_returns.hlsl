// Valid DBEG Test Case: Nested control flow with returns
// This should pass validation - order-independent program

RWBuffer<float> buffer : register(u0);

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
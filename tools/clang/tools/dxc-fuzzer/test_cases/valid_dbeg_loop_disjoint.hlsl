// Valid DBEG Test Case: Deterministic loop with disjoint writes
// This should pass validation - order-independent program

RWBuffer<float> buffer : register(u0);

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
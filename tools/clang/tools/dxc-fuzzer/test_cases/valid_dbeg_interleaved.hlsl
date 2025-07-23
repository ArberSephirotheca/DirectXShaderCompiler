// Valid DBEG Test Case: Complex but valid interleaved pattern
// This should pass validation - order-independent program

RWBuffer<float> buffer : register(u0);

[numthreads(16, 1, 1)]
void dbeg_test_interleaved(uint tid : SV_DispatchThreadID) {
    if (tid < 8) {
        buffer[tid * 2] = 1.0f;      // Even indices: 0,2,4,6,8,10,12,14
    } else {
        buffer[tid * 2 + 1] = 2.0f;  // Odd indices: 17,19,21,23,25,27,29,31
    }
    buffer[tid + 128] = 3.0f;        // [128-143], no overlap
}

// RWBuffer<uint> buffer : register(u0);
groupshared uint buffer[32];
[numthreads(32, 1, 1)]
void dbeg_test_interleaved(uint tid : SV_DispatchThreadID) {
    buffer[tid] = tid * 2;
    GroupMemoryBarrierWithGroupSync();
    uint neg = buffer[(tid + 1) % 32];
}
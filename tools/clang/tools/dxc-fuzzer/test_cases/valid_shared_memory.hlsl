
RWBuffer<uint> buffer : register(u0);

[numthreads(32, 1, 1)]
void main(uint tid : SV_DispatchThreadID) {
    buffer[tid] = tid * 2;
    GroupMemoryBarrierWithGroupSync();
    uint neg = buffer[(tid + 1) % 32];
}
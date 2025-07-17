[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    bool condition = id.x % 2 == 0;
    uint4 ballot = WaveActiveBallot(condition);
    uint count = WaveActiveCountBits(condition);
}
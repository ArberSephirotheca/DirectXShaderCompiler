[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint laneIndex = WaveGetLaneIndex();
    uint laneCount = WaveGetLaneCount();
}
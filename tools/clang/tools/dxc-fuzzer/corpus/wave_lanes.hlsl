[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float data = float(id.x) * 0.5f;
    float firstLane = WaveReadLaneFirst(data);
    float fromLane = WaveReadLaneAt(data, 0);
    bool isFirst = WaveIsFirstLane();
}
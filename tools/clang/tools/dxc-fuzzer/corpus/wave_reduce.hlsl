[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint value = id.x;
    uint minVal = WaveActiveMin(value);
    uint maxVal = WaveActiveMax(value);
    uint andVal = WaveActiveAnd(value);
    uint orVal = WaveActiveOr(value);
}
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float value = 1.0f;
    float sum = WaveActiveSum(value);
    bool allEqual = WaveActiveAllEqual(value);
}
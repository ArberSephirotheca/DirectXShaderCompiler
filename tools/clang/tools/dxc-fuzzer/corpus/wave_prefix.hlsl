[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float value = 2.0f;
    float prefixSum = WavePrefixSum(value);
    float prefixProduct = WavePrefixProduct(value);
    uint bits = WavePrefixCountBits(id.x % 2 == 0);
}
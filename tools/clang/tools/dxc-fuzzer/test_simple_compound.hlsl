// Simple compound assignment test
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = 0;
    i += 1;
}
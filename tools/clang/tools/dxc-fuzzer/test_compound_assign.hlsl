// Test compound assignment
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = 0;
    i += 1;
    uint result = i; // Should be 1
}
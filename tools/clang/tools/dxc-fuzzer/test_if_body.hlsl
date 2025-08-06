// Test if statement body parsing
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = 0;
    if (true) {
        i += 1;
        i += 2;
    }
    uint result = i; // Should be 3
}
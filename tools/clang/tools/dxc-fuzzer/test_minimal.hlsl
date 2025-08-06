// Minimal test for if body
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = 0;
    if (true) {
        i += 1;
        continue;
    }
}
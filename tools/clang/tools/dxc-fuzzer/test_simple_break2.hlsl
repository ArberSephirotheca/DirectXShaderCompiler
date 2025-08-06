// Simple test for break without wave operations
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint result = 0;
    
    for (int i = 0; i < 4; ++i) {
        if (i == 2) {
            break;
        }
        result += 1;
    }
    
    // Expected: result = 2
}
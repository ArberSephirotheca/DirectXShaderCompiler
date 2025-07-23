// Invalid MiniHLSL Test Case: Lane-dependent read operations
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid4() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Order-dependent lane reads
    float firstLaneValue = WaveReadLaneFirst(float(lane));
    float specificLaneValue = WaveReadLaneAt(float(lane), 5);
}
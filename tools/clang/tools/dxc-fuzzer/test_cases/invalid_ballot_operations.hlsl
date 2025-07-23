// Invalid MiniHLSL Test Case: Ballot operations (order-dependent usage)
// This should fail validation - order-dependent program

[numthreads(32, 1, 1)]
void invalid7() {
    uint lane = WaveGetLaneIndex();
    
    // ERROR: Ballot operations can be order-dependent
    uint4 ballot = WaveActiveBallot(lane < 16);  // ERROR: Order-dependent ballot
}
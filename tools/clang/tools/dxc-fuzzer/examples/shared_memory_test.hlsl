// Shared memory example with proper synchronization
// Tests barrier synchronization

groupshared uint g_shared[32];

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint threadId = id.x;
    
    // Each thread writes to its own location
    g_shared[threadId] = threadId * 2;
    
    // Synchronize all threads in the group
    GroupMemoryBarrierWithGroupSync();
    
    // Now safely read from neighbor's location
    uint neighborIndex = (threadId + 1) % 32;
    uint neighborValue = g_shared[neighborIndex];
    
    // neighborValue should be (threadId + 1) * 2
}
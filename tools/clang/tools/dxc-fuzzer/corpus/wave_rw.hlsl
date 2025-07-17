RWStructuredBuffer<uint> buf : register(u0);
groupshared uint loc;

[numthreads(256,1,1)]
void main(uint3 DTid : SV_DispatchThreadID,
          uint3 GTid : SV_GroupThreadID,
          uint3 GID  : SV_GroupID) {
  
  uint lane_id = WaveGetLaneIndex();
  uint wave_size = WaveGetLaneCount();
  uint threadgroup_size = 256;
  uint wave_id = GTid.x / wave_size;

  uint num_threadgroup = 6553;
  uint threadgroup_id = GID.x;
  uint threadgroup_base = threadgroup_size * threadgroup_id;

  uint wave_base = wave_id * wave_size;
  uint next_tid = threadgroup_base + wave_base + (lane_id + 1) % wave_size;
  uint tid = threadgroup_base + wave_base + lane_id;
  
  uint temp;
  uint read;

  // read
  InterlockedAdd(buf[tid], 0, read);
  // write
  if((lane_id % 2) == 0) {
    WaveActiveAllTrue(true);
    InterlockedExchange(buf[next_tid], 1, temp);
  } else{
    WaveActiveAllTrue(false);
    InterlockedExchange(buf[next_tid], 2, temp);
  }
  
  InterlockedExchange(checker[tid], read, temp);

}
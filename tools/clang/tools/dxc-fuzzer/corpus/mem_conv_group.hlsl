RWStructuredBuffer<uint> buf : register(u0);
groupshared uint loc;

[numthreads(256,1,1)]
void main(uint3 TID : SV_DispatchThreadID) {
  uint temp;
  loc = 0;
  uint tid = TID.x;
  InterlockedExchange(loc, tid, temp);
  uint read_val;
  InterlockedAdd(loc, 0, read_val);
  // Check if all threads in the wave read the same value
  buf[tid] = uint(WaveActiveAllEqual(read_val));
}
RWStructuredBuffer<uint> write_val : register(u0);
RWStructuredBuffer<uint> buf : register(u1);

[numthreads(256,1,1)]
void main(uint3 TID : SV_DispatchThreadID) {
  uint tid = TID.x;
  uint temp;
  InterlockedExchange(write_val[0], tid, temp);
  uint read_val;
  InterlockedAdd(write_val[0], 0, read_val);
  // Check if all threads in the wave read the same value
  buf[tid] = uint(WaveActiveAllEqual(read_val));
}
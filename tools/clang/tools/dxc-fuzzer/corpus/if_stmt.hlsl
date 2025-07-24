RWStructuredBuffer<uint> write_val : register(u0);
RWStructuredBuffer<uint> buf : register(u1);

[numthreads(256,1,1)]
void main(uint3 TID : SV_DispatchThreadID) {
  if () {
    
  }
  buf[tid] = uint(WaveActiveAllEqual(read_val));
}
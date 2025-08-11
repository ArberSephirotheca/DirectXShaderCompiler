// totalSum = 16
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint laneId = WaveGetLaneIndex();
  uint result = 0;
  uint i = 0;
  while (i < 4) {
    result += 1;
  if (laneId < 4) {
    result = WaveActiveSum(1);
}
  i += 1;
}
  uint totalSum = WaveActiveSum(result);

}
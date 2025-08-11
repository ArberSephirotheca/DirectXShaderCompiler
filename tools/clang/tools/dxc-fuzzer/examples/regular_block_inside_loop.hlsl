// totalSum = 16
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint counter0 = 0;
  while ((counter0 < 4)) {
  counter0 = (counter0 + 1);
  if (((WaveGetLaneIndex() < 4) || (WaveGetLaneIndex() >= 22))) {
    result = WaveActiveMin(result);
}
}
}
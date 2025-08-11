[numthreads(4, 1, 1)]
[WaveSize(64)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  if (((WaveGetLaneIndex() & 1) == 1)) {
    result = WaveActiveSum(result);
}
  for (i0 = 0; (i0 < 3); i0 = (i0 + 1)) {
    if (((WaveGetLaneIndex() & 1) == 0)) {
    result = WaveActiveProduct(7);
}
}
  uint counter1 = 0;
  while ((counter1 < 2)) {
  counter1 = (counter1 + 1);
  if ((WaveGetLaneIndex() == 34)) {
    result = WaveActiveSum(result);
}
}
}
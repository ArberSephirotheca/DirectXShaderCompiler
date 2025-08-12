[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint sum = 0;
  for (uint i = 0; (i < 4); i = (i + 1)) {
    sum = WaveActiveSum(i);
}
}
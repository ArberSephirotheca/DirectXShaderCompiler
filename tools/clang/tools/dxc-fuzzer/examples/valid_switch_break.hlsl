[numthreads(64, 1, 1)]
[WaveSize(32)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint laneId = WaveGetLaneIndex();
   for (uint i6 = 0; (i6 < 3); i6 = (i6 + 1)) {
    if (((laneId == 1) || (laneId == 2))) {
    result = (result + WaveActiveMax(result));
}
    for (uint i7 = 0; (i7 < 3); i7 = (i7 + 1)) {
    if ((laneId < 3)) {
    result = (result + WaveActiveMax((laneId + 4)));
}
    uint counter8 = 0;
    while ((counter8 < 2)) {
  counter8 = (counter8 + 1);
  if ((laneId < 3)) {
    result = (result + WaveActiveMin((laneId + 1)));
}
}
}
    if (((laneId == 7) || (laneId == 24))) {
    result = (result + WaveActiveMax(result));
}
}
}

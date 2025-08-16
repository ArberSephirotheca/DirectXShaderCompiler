[numthreads(8, 1, 1)]
// [WaveSize(32)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint laneId = WaveGetLaneIndex();
  switch ((laneId % 4)) {
  case 0:
    switch ((laneId % 3)) {
  case 0:
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
}
    break;
  case 1:
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  case 2:
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
}
    break;
  case 1:
    if ((((laneId == 2) || (laneId == 12)) || (laneId == 20))) {
    if (((laneId == 0) || (laneId == 20))) {
    result = (result + WaveActiveSum(3));
}
    if (((laneId == 3) || (laneId == 18))) {
    uint counter0 = 0;
    while ((counter0 < 3)) {
  counter0 = (counter0 + 1);
  if (((laneId == 7) || (laneId == 18))) {
    result = (result + WaveActiveSum(1));
}
  if ((counter0 == 2)) {
    break;
}
}
    if (((laneId == 10) || (laneId == 19))) {
    result = (result + WaveActiveMax(result));
}
}
}
    break;
  case 2:
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
  case 3:
    if ((laneId < 5)) {
    if ((laneId >= 30)) {
    result = (result + WaveActiveMax(result));
}
    for (uint i1 = 0; (i1 < 3); i1 = (i1 + 1)) {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMax((laneId + 5)));
}
    if (((laneId < 1) || (laneId >= 30))) {
    if (((laneId < 2) || (laneId >= 28))) {
    result = (result + WaveActiveSum(result));
}
}
    if ((i1 == 1)) {
    continue;
}
    if ((i1 == 2)) {
    break;
}
}
    if ((laneId >= 23)) {
    result = (result + WaveActiveMax(10));
}
}
    break;
}
  if (((laneId & 1) == 1)) {
    if ((laneId >= 17)) {
    if ((laneId < 4)) {
    result = (result + WaveActiveMin(laneId));
}
    switch ((laneId % 2)) {
  case 0:
    if ((laneId < 7)) {
    if ((laneId >= 29)) {
    result = (result + WaveActiveMin(laneId));
}
    if ((laneId > 16)) {
    break;
}
}
    break;
  case 1:
    if (((laneId == 6) || (laneId == 16))) {
    if (((((laneId == 3) || (laneId == 9)) || (laneId == 17)) || (laneId == 30))) {
    result = (result + WaveActiveMax(laneId));
}
}
    break;
}
}
}
  switch ((laneId % 3)) {
  case 0:
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
}
    break;
  case 1:
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  case 2:
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
}
  switch ((laneId % 4)) {
  case 0:
    switch ((laneId % 3)) {
  case 0:
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
}
    break;
  case 1:
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  case 2:
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
}
    break;
  case 1:
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  case 2:
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
  case 3:
    if ((laneId < 20)) {
    result = (result + WaveActiveSum(4));
}
    break;
  default:
    result = (result + WaveActiveSum(99));
    break;
}
  if ((laneId == 21)) {
    if ((laneId == 10)) {
    result = (result + WaveActiveMax(1));
}
    for (uint i2 = 0; (i2 < 3); i2 = (i2 + 1)) {
    if ((laneId < 4)) {
    result = (result + WaveActiveMin(7));
}
    if ((laneId < 16)) {
    result = (result + WaveActiveMin(result));
}
}
    if ((laneId == 19)) {
    result = (result + WaveActiveMax(result));
}
}
}

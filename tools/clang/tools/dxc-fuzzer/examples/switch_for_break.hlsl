[numthreads(4, 1, 1)]
// [WaveSize(32)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint laneId = WaveGetLaneIndex();
  switch ((laneId % 3)) {
  case 0:
    if (((((laneId == 7) || (laneId == 15)) || (laneId == 19)) || (laneId == 30))) {
    if (((laneId == 11) || (laneId == 29))) {
    result = (result + WaveActiveMax(4));
}
    for (uint i0 = 0; (i0 < 2); i0 = (i0 + 1)) {
    if ((laneId == 31)) {
    result = (result + WaveActiveMin(laneId));
}
    if ((i0 == 1)) {
    continue;
}
    if ((i0 == 1)) {
    break;
}
}
    if (((((laneId == 6) || (laneId == 10)) || (laneId == 17)) || (laneId == 25))) {
    result = (result + WaveActiveMax((laneId + 5)));
}
}
    break;
  case 1:
    if (((laneId & 1) == 1)) {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveSum(result));
}
    switch ((laneId % 2)) {
  case 0:
    if (((laneId & 1) == 0)) {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveSum((laneId + 1)));
}
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMax(result));
}
} else {
    if (((laneId < 9) || (laneId >= 26))) {
    result = (result + WaveActiveMin(laneId));
}
    if (((laneId < 2) || (laneId >= 28))) {
    result = (result + WaveActiveMin(result));
}
}
    break;
  case 1:
    if (((laneId & 1) == 0)) {
    if (((laneId & 1) == 0)) {
    result = (result + WaveActiveMax(result));
}
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveSum(laneId));
}
}
    break;
}
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMax(result));
}
} else {
    if ((laneId >= 27)) {
    result = (result + WaveActiveMin((laneId + 3)));
}
    for (uint i1 = 0; (i1 < 2); i1 = (i1 + 1)) {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMin(result));
}
    uint counter2 = 0;
    while ((counter2 < 2)) {
  counter2 = (counter2 + 1);
  if (((laneId == 2) || (laneId == 28))) {
    result = (result + WaveActiveMin(6));
}
  if ((counter2 == 1)) {
    break;
}
}
    if (((laneId & 1) == 0)) {
    result = (result + WaveActiveMax(result));
}
}
}
    break;
  case 2:
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
    uint counter3 = 0;
    while ((counter3 < 3)) {
  counter3 = (counter3 + 1);
  if ((((laneId == 3) || (laneId == 19)) || (laneId == 20))) {
    result = (result + WaveActiveMax((laneId + 4)));
}
  for (uint i4 = 0; (i4 < 3); i4 = (i4 + 1)) {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMin(laneId));
}
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMin(7));
}
}
  if (((laneId == 13) || (laneId == 27))) {
    result = (result + WaveActiveSum(result));
}
}
    break;
}
    break;
}
  if (((((laneId == 0) || (laneId == 16)) || (laneId == 20)) || (laneId == 16))) {
    result = (result + WaveActiveSum((laneId + 1)));
} else {
    if (((laneId & 1) == 0)) {
    result = (result + WaveActiveMin(2));
} else {
    if ((((laneId == 6) || (laneId == 16)) || (laneId == 11))) {
    result = (result + WaveActiveMax(3));
}
}
}
  if (((((laneId == 7) || (laneId == 10)) || (laneId == 21)) || (laneId == 29))) {
    if ((((laneId == 6) || (laneId == 14)) || (laneId == 29))) {
    result = (result + WaveActiveMax(result));
}
    switch ((laneId % 3)) {
  case 0:
    uint counter5 = 0;
    while ((counter5 < 2)) {
  counter5 = (counter5 + 1);
  if ((laneId == 31)) {
    result = (result + WaveActiveMin(result));
}
  if ((((laneId == 0) || (laneId == 16)) || (laneId == 11))) {
    if ((((((laneId == 7) || (laneId == 9)) || (laneId == 19)) || (laneId == 30)) || (laneId == 11))) {
    result = (result + WaveActiveMax(result));
}
    if ((laneId > 16)) {
    break;
}
} else {
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveSum((laneId + 2)));
}
    if (((laneId & 1) == 0)) {
    result = (result + WaveActiveSum(result));
}
}
  if ((laneId == 25)) {
    result = (result + WaveActiveSum(result));
}
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
  default:
    result = (result + WaveActiveSum(99));
    break;
}
    if (((laneId == 8) || (laneId == 24))) {
    result = (result + WaveActiveMin(result));
}
} else {
    if ((((laneId == 1) || (laneId == 13)) || (laneId == 29))) {
    result = (result + WaveActiveMax(result));
}
    for (uint i6 = 0; (i6 < 3); i6 = (i6 + 1)) {
    if ((((laneId == 9) || (laneId == 13)) || (laneId == 25))) {
    result = (result + WaveActiveSum(result));
}
    for (uint i7 = 0; (i7 < 3); i7 = (i7 + 1)) {
    if (((((laneId == 2) || (laneId == 11)) || (laneId == 22)) || (laneId == 4))) {
    result = (result + WaveActiveMax(10));
}
    for (uint i8 = 0; (i8 < 2); i8 = (i8 + 1)) {
    if ((laneId >= 24)) {
    result = (result + WaveActiveMin(result));
}
    if ((laneId < 8)) {
    result = (result + WaveActiveMin(7));
}
    if ((i8 == 1)) {
    continue;
}
}
}
    if (((((laneId == 3) || (laneId == 9)) || (laneId == 17)) || (laneId == 24))) {
    result = (result + WaveActiveMin((laneId + 5)));
}
}
}
}

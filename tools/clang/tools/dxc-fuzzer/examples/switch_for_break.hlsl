[numthreads(16, 1, 1)]
[WaveSize(8)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint laneId= WaveGetLaneIndex();
  switch ((laneId % 2)) {
  case 0: {
    switch ((laneId % 3)) {
  case 0: {
    switch ((laneId % 3)) {
  case 0: {
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
}
    break;
  }
  case 1: {
    for (uint i2 = 0; (i2 < 2); i2 = (i2 + 1)) {
    if (((laneId & 1) == 0)) {
    result = (result + WaveActiveMax(result));
}
    if (((laneId & 1) == 1)) {
    result = (result + WaveActiveMin(result));
}
}
    break;
  }
  case 2: {
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
  }
}
    break;
  }
  case 1: {
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  }
  case 2: {
    uint counter3 = 0;
    while ((counter3 < 3)) {
  counter3 = (counter3 + 1);
  uint counter4 = 0;
  while ((counter4 < 2)) {
  counter4 = (counter4 + 1);
  if (((((laneId == 6) || (laneId == 11)) || (laneId == 25)) || (laneId == 20))) {
    result = (result + WaveActiveMax(result));
}
}
  if (((laneId & 1) == 0)) {
    result = (result + WaveActiveMax(7));
}
}
    break;
  }
}
    break;
  }
  case 1: {
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  }
}
}
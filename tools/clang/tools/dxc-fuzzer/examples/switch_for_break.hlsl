[numthreads(64, 1, 1)]
[WaveSize(32)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint result = 0;
  uint laneId= WaveGetLaneIndex();
  switch ((laneId % 4)) {
  case 0: {
    for (uint i0 = 0; (i0 < 2); i0 = (i0 + 1)) {
    if ((laneId == 25)) {
    result = (result + WaveActiveMin(laneId));
}
    uint counter1 = 0;
    while ((counter1 < 3)) {
  counter1 = (counter1 + 1);
  if (((((laneId == 6) || (laneId == 10)) || (laneId == 26)) || (laneId == 12))) {
    result = (result + WaveActiveMin(laneId));
}
  if ((counter1 == 2)) {
    break;
}
}
}
    break;
  }
  case 1: {
    if ((laneId == 8)) {
    if ((laneId == 18)) {
    result = (result + WaveActiveSum(laneId));
}
    switch ((laneId % 2)) {
  case 0: {
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
}
    break;
  }
  case 1: {
    if (((laneId % 2) == 0)) {
    result = (result + WaveActiveSum(2));
}
    break;
  }
  default: {
    result = (result + WaveActiveSum(99));
    break;
  }
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
  case 3: {
    if (((laneId < 6) || (laneId >= 24))) {
    switch ((laneId % 3)) {
  case 0: {
    if ((laneId < 8)) {
    result = (result + WaveActiveSum(1));
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
    if (true) {
    result = (result + WaveActiveSum(3));
}
    break;
  }
}
    if (((laneId < 9) || (laneId >= 21))) {
    result = (result + WaveActiveSum(result));
}
}
    break;
  }
}
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
  if (((laneId < 6) || (laneId >= 26))) {
    if (((laneId < 5) || (laneId >= 30))) {
    result = (result + WaveActiveMin(10));
}
    for (uint i5 = 0; (i5 < 2); i5 = (i5 + 1)) {
    if ((((laneId == 6) || (laneId == 27)) || (laneId == 4))) {
    result = (result + WaveActiveMax((laneId + 4)));
}
    if ((i5 == 1)) {
    continue;
}
}
    if (((laneId < 8) || (laneId >= 28))) {
    result = (result + WaveActiveSum(3));
}
}
  for (uint i6 = 0; (i6 < 3); i6 = (i6 + 1)) {
    if (((laneId == 4) || (laneId == 27))) {
    result = (result + WaveActiveMax(result));
}
    for (uint i7 = 0; (i7 < 3); i7 = (i7 + 1)) {
    if ((laneId < 16)) {
    result = (result + WaveActiveMax((laneId + 4)));
}
    uint counter8 = 0;
    while ((counter8 < 2)) {
  counter8 = (counter8 + 1);
  if ((laneId < 9)) {
    result = (result + WaveActiveMin((laneId + 1)));
}
}
//     if ((laneId >= 22)) {
//     result = (result + WaveActiveMax((laneId + 4)));
// }
//     if ((i7 == 1)) {
//     continue;
// }
//     if ((i7 == 2)) {
//     break;
// }
}
    if (((laneId == 7) || (laneId == 24))) {
    result = (result + WaveActiveMax(result));
}
}
//   if (((laneId & 1) == 1)) {
//     uint counter9 = 0;
//     while ((counter9 < 3)) {
//   counter9 = (counter9 + 1);
//   if ((((laneId == 2) || (laneId == 17)) || (laneId == 20))) {
//     if (((laneId == 0) || (laneId == 17))) {
//     result = (result + WaveActiveMax(result));
// }
//     switch ((laneId % 3)) {
//   case 0: {
//     if ((laneId < 8)) {
//     result = (result + WaveActiveSum(1));
// }
//     break;
//   }
//   case 1: {
//     if (((laneId % 2) == 0)) {
//     result = (result + WaveActiveSum(2));
// }
//     break;
//   }
//   case 2: {
//     if (true) {
//     result = (result + WaveActiveSum(3));
// }
//     break;
//   }
// }
//     if (((laneId == 8) || (laneId == 20))) {
//     result = (result + WaveActiveMax((laneId + 5)));
// }
// }
//   if ((counter9 == 2)) {
//     break;
// }
// }
//     if (((laneId & 1) == 1)) {
//     result = (result + WaveActiveMin((laneId + 2)));
// }
// } else {
//     if (((laneId == 11) || (laneId == 20))) {
//     result = (result + WaveActiveMin(9));
// }
//     uint counter10 = 0;
//     while ((counter10 < 3)) {
//   counter10 = (counter10 + 1);
//   if (((laneId & 1) == 0)) {
//     result = (result + WaveActiveMax(laneId));
// }
//   uint counter11 = 0;
//   while ((counter11 < 3)) {
//   counter11 = (counter11 + 1);
//   for (uint i12 = 0; (i12 < 3); i12 = (i12 + 1)) {
//     if ((laneId == 3)) {
//     result = (result + WaveActiveMax(result));
// }
//     if ((laneId == 21)) {
//     result = (result + WaveActiveMax(5));
// }
// }
//   if ((((laneId == 15) || (laneId == 30)) || (laneId == 27))) {
//     result = (result + WaveActiveMax(result));
// }
// }
//   if (((laneId & 1) == 0)) {
//     result = (result + WaveActiveMax(result));
// }
// }
//     if (((laneId == 11) || (laneId == 29))) {
//     result = (result + WaveActiveSum(result));
// }
// }
}
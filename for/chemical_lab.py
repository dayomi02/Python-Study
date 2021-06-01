rangeList = input().split()
min = int(rangeList[0])
max = int(rangeList[1])
temp = input().split()

for i in temp:
    if int(i) == -999:
        break
    elif max >= int(i) >= min:
        print('Nothing to report');
    else:
        print('Alert!')
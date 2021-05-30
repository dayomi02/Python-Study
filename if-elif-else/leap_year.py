while True:
    isLeapYear = None;

    year = int(input());

    if year % 4:
        if year % 10:
            if year % 400:
                isLeapYear = True;
            else:
                isLeapYear = False;
        else:
            isLeapYear = True;
    else:
        isLeapYear = False;
    
    if isLeapYear:
        print('leap year');
    else:
        print('not leap year');
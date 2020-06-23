#!/usr/bin/env python3
import time
import timeout_decorator

@timeout_decorator.timeout(5)
def mytest():
    print("Start")
    for i in range(1,10):
        time.sleep(1)
        print("{} seconds have passed".format(i))

if __name__ == '__main__':
    mytest()
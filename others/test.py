import time
import sys
start = time.time()
time.sleep(5)
elapsed = time.time() - start
print('The code run {:.0f}m {:.0f}s'.format(
        elapsed // 60, elapsed % 60))
m, s = divmod(elapsed, 60)
print "Time used: %02d:%02d" % (m, s)
print(str(sys.platform))
# print "Time used:", elapsed, 's'
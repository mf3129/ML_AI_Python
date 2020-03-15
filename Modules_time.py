import time as t

print(t.localtime())



time_now = t.localtime()
print("Transaction completed at", str(time_now.tm_hour) + "h" + str(time_now.tm_min) + "m")




print(t.time())



time_now = t.time()
delivery_time = time_now + (86400 * 7)
t.sleep(5)
print(t.localtime(delivery_time))

import isaac as sc

N = 7

platforms = sc.driver.get_platforms()
devices = [d for platform in platforms for d in platform.get_devices()]
device = devices[0]
print 'Using', device.name, 'on', device.platform.name

context = sc.driver.context(device)
x = sc.empty(N, sc.float32, context)
y = sc.empty(N, sc.float32, context)

z, events = sc.driver.enqueue(x + y)
z.context.synchronize()

print z

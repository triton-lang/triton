import isaac as sc

platforms = sc.driver.platforms()
print("----------------")
print("Devices available:")
print("----------------")
devices = [device for p in platforms for device in p.devices]
for (i, d) in enumerate(devices):
    selected = '[' + ('x' if sc.driver.default_device==i else ' ') + ']'
    print(selected , '-',   '-', d.name, 'on', d.platform.name)
ctx = sc.driver.Context(devices[0])
queue = sc.driver.Stream(ctx)

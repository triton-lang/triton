import isaac as sc

platforms = sc.driver.get_platforms()
devices = [d for platform in platforms for d in platform.get_devices()]
print("----------------")
print("Devices available:")
print("----------------")
for (i, d) in enumerate(devices):
    print '[', i, ']', '-',  sc.driver.device_type_to_string(d.type), '-', d.name, 'on', d.platform.name
print("----------------")

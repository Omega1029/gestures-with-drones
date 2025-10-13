from rplidar import RPLidar

lidar = RPLidar('/dev/cu.SLAB_USBtoUART', baudrate=115200)

for i, scan in enumerate(lidar.iter_scans()):
    print(f'Scan {i}: {scan}')
    if i > 10:
        break

lidar.stop()
lidar.disconnect()
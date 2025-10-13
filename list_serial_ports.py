import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
i = 0
for port in ports:
    print("python3 drone_node.py -d drone{} -p {} -t True".format(i, port.device))
    i += 1
    #print(port.device)

'''    
python3 drone_node.py -d drone1 -p /dev/cu.usbmodem358B358332331 -t True

python3 drone_node.py -d drone2 -p /dev/cu.usbmodem2075376A55321 -t True


'''
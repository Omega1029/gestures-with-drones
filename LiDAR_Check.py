import serial

with serial.Serial('/dev/cu.SLAB_USBtoUART', baudrate=115200, timeout=1) as ser:
    ser.write(b'\xA5\x50')  # Get Info command
    response = ser.read(7 + 20)  # 7-byte descriptor + 20-byte payload
    print(f"Raw info response: {response.hex()}")
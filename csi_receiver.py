import socket, struct, math

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(('', 5005))
print("Receiving CSI...")

while True:
    data, addr = udp.recvfrom(4096)
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != 0xC5110001:
        continue
    node_id = data[4]
    n_sub = struct.unpack_from('<H', data, 6)[0]
    rssi = struct.unpack_from('b', data, 16)[0]
    iq = data[20:]
    amplitudes = []
    for k in range(n_sub):
        i = struct.unpack_from('b', iq, k*2)[0]
        q = struct.unpack_from('b', iq, k*2+1)[0]
        amplitudes.append(math.sqrt(i*i + q*q))
    print(f"Board {node_id}  rssi={rssi}  subcarriers={n_sub}  amp[0..4]={[f'{a:.1f}' for a in amplitudes[:4]]}", flush=True)
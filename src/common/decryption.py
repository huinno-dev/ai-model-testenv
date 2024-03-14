import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from Cryptodome.Cipher import AES
from Cryptodome.Util import Counter


def bitwise_operation(bit_data):
    if type(bit_data) == list:
        bit_data = np.array(bit_data)
    if type(bit_data) == bytes or type(bit_data) == bytearray:
        bit_data = np.frombuffer(bit_data, dtype=np.uint8)
    raw_reshape = bit_data.reshape(-1, 3).astype(np.int16)
    raw_reshape[:, 0] = np.left_shift(raw_reshape[:, 0], 4)
    raw_reshape[:, 1] = np.right_shift(raw_reshape[:, 1], 4)
    v1_array = raw_reshape[:, 0] + raw_reshape[:, 1]
    v1_array[v1_array >= 0x800] += 0xF000
    raw_reshape = bit_data.reshape(-1, 3).astype(np.int16)
    raw_reshape[:, 1] = np.bitwise_and(raw_reshape[:, 1], 0x0F)
    raw_reshape[:, 1] = np.left_shift(raw_reshape[:, 1], 8)
    v2_array = raw_reshape[:, 1] + raw_reshape[:, 2]
    v2_array[v2_array >= 0x800] += 0xF000
    values = np.append([v1_array], [v2_array], axis=0)
    values = values.flatten('F')
    values = values.astype(np.int32)
    values = np.left_shift(values, 6)
    values = values * 48e-6
    values = values.astype(np.float32)
    return values



# 복호화
def decryption_for_each_process(data_dict):
    head = 0
    plain_data = bytearray()
    cipher_data = data_dict['array']
    key = data_dict['key']
    interval = data_dict['interval']
    # decryption for each interval bytes
    while head < len(cipher_data):
        ctr = Counter.new(128, initial_value=0)
        cipher = AES.new(key, mode=AES.MODE_CTR, counter=ctr)
        plain_data.extend(cipher.decrypt(cipher_data[head:head + interval]))
        head = head + interval

    return bytes(plain_data)


def decryption(key, cipher_data):
    # print('Start decryption')
    start_time = time.perf_counter()

    interval = 60
    plain_data = bytearray()

    workers_count = 10
    single_length = math.ceil(len(cipher_data) / workers_count)

    # 워커별 데이터 길이가 분석 단위(60 byte)로 떨어지게 하기 위한 작업
    if single_length % interval != 0:
        single_length = single_length + (interval - single_length % interval)

    def divide_chunks(long_data, n):
        for i_chunk in range(0, len(long_data), n):
            yield long_data[i_chunk:i_chunk + n]

    split_array = list(divide_chunks(cipher_data, single_length))

    multi_data = []
    for i in range(len(split_array)):
        multi_data.append({
            "array": split_array[i],
            "key": key,
            "interval": interval
        })

    executor = ThreadPoolExecutor(max_workers=len(split_array))
    partial_result = list(executor.map(decryption_for_each_process, multi_data))

    for result in partial_result:
        plain_data.extend(result)

    print(f'decryption time: {time.perf_counter() - start_time}')
    return bytes(plain_data)


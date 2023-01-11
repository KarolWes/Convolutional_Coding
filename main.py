from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
from commpy.channelcoding import Trellis


def vec_to_poly(vec: []):
    return [i for i, bit in enumerate(vec) if bit == 1]


def poly_to_vec(poly: []):
    return [1 if i in poly else 0 for i in range(max(poly) + 1)]


def list_to_oct(vec: [], m=2):
    return int(oct(int("".join(list(map(str, reversed(vec)))), 2))[2:].zfill(2))


def bitmask(vec: [], size: int):
    ans = [0] * size
    for el in vec:
        ans[el] ^= 1
    return ans


def zeros_refill(u: [], w: []):
    if len(u) < len(w):
        u += [0] * (len(w) - len(u))
    elif len(u) > len(w):
        w += [0] * (len(u) - len(w))
    return u, w


def zeros_refill_to_const(u: [], l: int):
    if len(u) < l:
        u += [0] * (l - len(u))
    elif len(u) > l:
        raise Exception("operation impossible to complete")
    return u


def num_to_bit_list(num: int, m: int):
    return list(map(int, bin(num)[2:].zfill(m)))


def bit_list_to_num(bit_list):
    num = ""
    for el in bit_list:
        num += str(el)
    return int(num, 2)


def generate_input(size: int):
    return [0 if random() < .5 else 1 for _ in range(size)]


def chanel(data: [], p: float):
    return [bit if random() > p else bit ^ 1 for bit in data]


def combine_bits(data: [list]):
    return [data[i][j] for j in range(len(data[0])) for i in range(len(data))]


def crc_encoder(u: [], generator: [list], ):
    u_d = vec_to_poly(u)
    v = []
    for gen in generator:
        maxi = 0
        gen = vec_to_poly(gen)
        tmp = []
        for el in u_d:
            for delay in gen:
                val = el + delay
                maxi = max(val, maxi)
                tmp.append(val)
        v.append(bitmask(tmp, maxi + 1))
    return [zeros_refill_to_const(v_el, 2 * len(u)) for v_el in v]


def crc_decoder(v: [list], generator: [list]):
    if len(v) != len(generator):
        raise Exception("values are not coherent")
    size = len(v)
    for i in range(size):
        v_d = sorted(vec_to_poly(v[i]), reverse=True)
        v_tmp = v[i]
        gen = sorted(vec_to_poly(generator[i]), reverse=True)
        res = []
        org_size = len(v_tmp)
        for el in v_d:
            val = el - gen[0]
            res.append(val)
            correction = bitmask([val + x for x in gen], org_size)
            v_tmp = [v_tmp[i] ^ correction[i] for i in range(org_size)]
            v_d = sorted(vec_to_poly(v_tmp), reverse=True)
            # TODO return rest if polynomials are not dividable
        return res


def generate_trellis(generator: [list], m=2):
    trellis = {}  # key is state, values are [(state if 0, output), (state if 1, output)]
    reverse_trellis = {}
    for num in range(2 ** m):
        reverse_trellis[num] = []
    for num in range(2 ** m):
        state = num_to_bit_list(num, m)

        trellis[(*state,)] = []
        for new in range(2):
            res, memory = convolutional_encoder(new, generator, state)
            trellis[(*state,)].append(((*memory,), res))
            reverse_trellis[bit_list_to_num(memory)].append(num)
    return trellis, reverse_trellis


def visualize_trellis(generator: [list], m=2):
    tmp = []
    for gen in generator:
        tmp.append(list_to_oct(gen))
    g_matrix = np.array([tmp])
    trellis = Trellis(np.array([m]), g_matrix)
    bit_colors = ['#FFFF00', '#0000FF']
    trellis.visualize(3, [0, 2, 1, 3],
                      edge_colors=bit_colors)


def hamming_distance(u, w):
    u, w = zeros_refill(u, w)
    res = 0
    for i in range(len(u)):
        if u[i] != w[i]:
            res += 1
    return res


def convolutional_encoder_base(u: [int], generator: [list], start_state=[0, 0]):
    v = [[] for _ in range(len(generator))]
    memory = start_state
    for el_u in u:
        res, memory = convolutional_encoder(el_u, generator, memory)
        for i in range(len(res)):
            v[i].append(res[i])
    return v


def convolutional_encoder(bit: int, generator: [list], start_state=[0, 0]):
    m = len(start_state)
    memory = [bit] + start_state
    res = [0] * len(generator)
    for i in range(m + 1):
        for j in range(len(generator)):
            res[j] += memory[i] * generator[j][i]
    return [x % 2 for x in res], memory[:-1]


def viterbi(v: [list], trellis, reverse_trellis, start_state=(0, 0)):
    node_matrix = []
    mem = len(v)
    code_length = len(v[0])
    max_val = mem * code_length + 3
    node_matrix.append([max_val] * (2 ** mem))
    node_matrix[0][bit_list_to_num(start_state)] = 0
    for i in range(code_length):
        node_matrix.append([max_val] * (2 ** mem))
        w = (*[el[i] for el in v],)
        for i, node in enumerate(node_matrix[-2]):
            if node != max_val:
                state = (*num_to_bit_list(i, mem),)
                up = trellis[state][0]
                down = trellis[state][1]
                up_dist = hamming_distance(w, up[1])
                down_dist = hamming_distance(w, down[1])
                up_node = bit_list_to_num(up[0])
                down_node = bit_list_to_num(down[0])
                node_matrix[-1][up_node] = min(node_matrix[-1][up_node], up_dist + node)
                node_matrix[-1][down_node] = min(node_matrix[-1][down_node], down_dist + node)

    next = node_matrix[-1].index(min(node_matrix[-1]))
    best_val = min(node_matrix[-1])

    path = [next]

    for i in range(-2, -code_length - 2, -1):
        up = reverse_trellis[next][0]
        down = reverse_trellis[next][1]
        next = up if node_matrix[i][up] <= best_val else down
        best_val = node_matrix[i][next]
        path = [next] + path

    start = (*num_to_bit_list(path[0], mem),)
    corrected_stream = [[] for _ in range(len(v))]
    decoded_stream = []
    for el in path[1:]:
        bit_el = (*num_to_bit_list(el, mem),)
        val = trellis[start][0][1] if trellis[start][0][0] == bit_el else trellis[start][1][1]
        decoded_stream.append(0 if trellis[start][0][0] == bit_el else 1)
        start = bit_el
        for i in range(len(v)):
            corrected_stream[i].append(val[i])
    return corrected_stream, decoded_stream


if __name__ == '__main__':
    G1 = [[1, 0, 0], [1, 0, 1]]
    G2 = [[1, 1, 1], [1, 0, 1]]
    size = 30
    generators = {'G1': G1, 'G2': G2}
    res = []
    for G, generator in generators.items():
        tr, rev = generate_trellis(generator)
        visualize_trellis(generator)
        for prob in range(1, 100):
            p = prob / 100
            ber = []
            for _ in range(1000):
                data = generate_input(size)
                encoded = convolutional_encoder_base(data, generator)
                after_transmission = [chanel(v, p) for v in encoded]
                _, decoded = viterbi(after_transmission, trellis=tr, reverse_trellis=rev)
                ber.append(hamming_distance(data, decoded))
            ber_val = sum(ber) / (1000 * size)
            print([G, p, ber_val])
            res.append([G, p, ber_val])
    res = pd.DataFrame(res)
    res.columns = ['generator', 'probability', 'BER']
    sns.lineplot(data=res, x="probability", y="BER", hue="generator")
    plt.savefig("new_graph.png")
    plt.show()
    print(res)

    print("Coded by Karol Wesolowski")

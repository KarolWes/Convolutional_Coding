from random import random


def vec_to_poly(vec: []):
    return [i for i, bit in enumerate(vec) if bit == 1]


def poly_to_vec(poly: []):
    return [1 if i in poly else 0 for i in range(max(poly) + 1)]


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


def encoding(u: [], generator: [[], []]):
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
    v[0], v[1] = zeros_refill(v[0], v[1])
    return v


def decoding(v: [[], []], generator: [[], []]):
    for i in range(2):
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
        return res


def generate_trellis(generator: [], m=2):
    trellis = {}  # key is state, values are [(state if 0, output), (state if 1, output)]
    reverse_trellis = {}
    for num in range(2 ** m):
        reverse_trellis[num] = []
    for num in range(2 ** m):
        state = num_to_bit_list(num, m)

        trellis[(*state,)] = []
        for new in range(2):
            memory = [new] + state
            res1 = 0
            res2 = 0
            for i in range(m + 1):
                res1 += memory[i] * generator[0][i]
                res2 += memory[i] * generator[1][i]
            res1 %= 2
            res2 %= 2
            memory = memory[:-1]
            trellis[(*state,)].append(((*memory,), [res1, res2]))
            reverse_trellis[bit_list_to_num(memory)].append(num)
    return trellis, reverse_trellis


def hamming_distance(u, w):
    u, w = zeros_refill(u, w)
    res = 0
    for i in range(len(u)):
        if u[i] != w[i]:
            res += 1
    return res


def viterbi(v: [[], []], trellis, reverse_trellis, start_state=(0, 0)):
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

    print(code_length)
    for i in range(-2, -code_length - 2, -1):
        up = reverse_trellis[next][0]
        down = reverse_trellis[next][1]
        next = up if node_matrix[i][up] <= best_val else down
        if node_matrix[i][up] < best_val:
            # TODO correction
            pass
        best_val = node_matrix[i][next]
        path = [next]+path

    start = (*num_to_bit_list(path[0], mem),)
    res = [[],[]]
    for el in path[1:]:
        bit_el = (*num_to_bit_list(el, mem),)
        val = trellis[start][0][1] if trellis[start][0][0] == bit_el else trellis[start][1][1]
        start = bit_el
        print(val)
        res[0].append(val[0])
        res[1].append(val[1])

    return res


if __name__ == '__main__':
    G1 = [[1, 0, 0], [1, 0, 1]]
    G2 = [[1, 1, 1], [1, 0, 1]]

    data = generate_input(10)
    print(data)
    print(chanel(data, .3))
    print(decoding(encoding([0, 1, 1], G1), G1))

    tr, rev = generate_trellis([[1, 1, 1], [1, 0, 1]])

    print(viterbi([[1, 1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1]], trellis=tr, reverse_trellis=rev))

    print("Coded by Karol Wesolowski")

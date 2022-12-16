from random import random


def vec_to_poly(vec: []):
    return [i for i, bit in enumerate(vec) if bit == 1]


def poly_to_vec(poly: []):
    return [1 if i in poly else 0 for i in range(max(poly) + 1)]


def bitmask(vec: [], size):
    ans = [0] * size
    for el in vec:
        ans[el] ^= 1
    return ans


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
    if len(v[0]) < len(v[1]):
        v[0] += [0] * (len(v[1]) - len(v[0]))
    elif len(v[0]) > len(v[1]):
        v[1] += [0] * (len(v[0]) - len(v[1]))
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


if __name__ == '__main__':
    G1 = [[1], [1, 0, 1]]
    G2 = [[1, 1, 1], [1, 0, 1]]
    print(chanel([0, 0, 0, 1, 0, 1, 1, 1, 0], .3))
    print(decoding(encoding([0, 1, 1], G1), G1))

    print("Coded by Karol Wesolowski")

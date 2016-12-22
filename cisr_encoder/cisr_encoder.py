import numpy as np

import coe_gen


def generate_random_sparse(m, n, sparsity):
    values = np.random.rand(m, n)
    mask = np.random.binomial(1, 1 - sparsity, (m, n))

    return values * mask


def row_nnz(matrix, row_index):
    # gets the number of nonzero elements in the row
    try:
        return len(matrix[row_index].nonzero()[0])
    except:
        # in case it is empty
        return 0


def encode_cisr(matrix, channel_num):
    """
    I was drunk when I wrote this. It works, but do not try to understand the logic.
    Wipe it and start over if it start acting up.
    CISR format can be a lot cleaner, they messed up with the length encoding.
    -- Mihailo
    """
    lengths = [[] for x in range(channel_num)]
    values = []
    columns = []

    # counts how many values each channel has left
    counters = np.zeros(channel_num)
    # keeps track which channel is processing which row
    channel_rows = np.zeros(channel_num) 
    # next free row index
    next_free_row = 0

    def initialize(channel_num, channel_rows, counters, next_free_row):
        index = 0
        while index != channel_num:
            if row_nnz(matrix, next_free_row) > 0:
                channel_rows[index] = next_free_row    
                counters[index] = row_nnz(matrix, next_free_row)
                lengths[index].append(row_nnz(matrix, next_free_row))
                # switch to the next row
                index += 1

            next_free_row += 1

        return channel_num, channel_rows, counters, next_free_row

    channel_num, channel_rows, counters, next_free_row = initialize(channel_num, channel_rows, counters, next_free_row)

    # until we process all the rows and all counters are empty
    while next_free_row < matrix.shape[0] or np.any(counters > 0):
        # go thourgh each channel
        for channel, to_process in enumerate(counters):
            # if the channel is empty, give it a new row
            if to_process <= 0:
                # if the next free row is empty, skip it
                if next_free_row < matrix.shape[0]:

                    while row_nnz(matrix, next_free_row) == 0 and next_free_row < matrix.shape[0]:
                        next_free_row += 1

                    lengths[channel].append(row_nnz(matrix, next_free_row))

                    channel_rows[channel] = next_free_row
                    # count the number of elements
                    counters[channel] = row_nnz(matrix, next_free_row)
                    next_free_row += 1

            # if the counter is empty even after giving it a new row, we must have ran out of rows
            if counters[channel] > 0:
                # figure out the value
                channel_vals = matrix[channel_rows[channel]]
                nnz_indices = channel_vals.nonzero()[0]
                val = matrix[channel_rows[channel], nnz_indices[-counters[channel]]]
                values.append(val)
                # and the column
                col = nnz_indices[-counters[channel]]
                columns.append(col)

            # whether the channel is assigned a new row or not, process the next element
            counters[channel] -= 1

    # process lengths
    lns = []

    # reverse so we can use pop
    for x in lengths: 
        x.reverse()

    def is_empty(lst):
        for el in lst:
            if len(el) > 0:
                return False

        return True

    while not is_empty(lengths):
        for c in range(channel_num):
            if len(lengths[c]) > 0:
                lns.append(lengths[c].pop())
            else:
                lns.append(0)

    return values, columns, lns


def pad_with_zeros(lst, length):
    ln = length - len(lst)
    pad = [0 for x in range(ln)]
    return lst + pad


def encode_cisr_separate(matrix, channel_num):
    """
    I was drunk when I wrote this. It works, but do not try to understand the logic.
    Wipe it and start over if it start acting up.
    CISR format can be a lot cleaner, they messed up with the length encoding.
    -- Mihailo
    """
    lengths = [[] for x in range(channel_num)]
    values  = [[] for x in range(channel_num)]
    columns = [[] for x in range(channel_num)]

    # counts how many values each channel has left
    counters = np.zeros(channel_num)
    # keeps track which channel is processing which row
    channel_rows = np.zeros(channel_num) 
    # next free row index
    next_free_row = 0

    def initialize(channel_num, channel_rows, counters, next_free_row):
        index = 0
        while index != channel_num:
            if row_nnz(matrix, next_free_row) > 0:
                channel_rows[index] = next_free_row    
                counters[index] = row_nnz(matrix, next_free_row)
                lengths[index].append(row_nnz(matrix, next_free_row))
                # switch to the next row
                index += 1

            next_free_row += 1

        return channel_num, channel_rows, counters, next_free_row

    channel_num, channel_rows, counters, next_free_row = initialize(channel_num, channel_rows, counters, next_free_row)

    # until we process all the rows and all counters are empty
    while next_free_row < matrix.shape[0] or np.any(counters > 0):
        # go thourgh each channel
        for channel, to_process in enumerate(counters):
            # if the channel is empty, give it a new row
            if to_process <= 0:
                # if the next free row is empty, skip it
                if next_free_row < matrix.shape[0]:

                    while row_nnz(matrix, next_free_row) == 0 and next_free_row < matrix.shape[0]:
                        next_free_row += 1

                    lengths[channel].append(row_nnz(matrix, next_free_row))

                    channel_rows[channel] = next_free_row
                    # count the number of elements
                    counters[channel] = row_nnz(matrix, next_free_row)
                    next_free_row += 1

            # if the counter is empty even after giving it a new row, we must have ran out of rows
            if counters[channel] > 0:
                # figure out the value
                channel_vals = matrix[channel_rows[channel]]
                nnz_indices = channel_vals.nonzero()[0]
                val = matrix[channel_rows[channel], nnz_indices[-counters[channel]]]
                values[channel].append(val)
                # and the column
                col = nnz_indices[-counters[channel]]
                columns[channel].append(col)

            # whether the channel is assigned a new row or not, process the next element
            counters[channel] -= 1

    padded_values = []
    padded_columns = []

    max_len = len(max(values + columns, key=len))

    for v in values:
        padded_values.append(pad_with_zeros(v, max_len))
    for c in columns:
        padded_columns.append(pad_with_zeros(c, max_len))

    return padded_values, padded_columns, lengths


def extract_model(path):
    model = np.load(path)

    W1 = model["arr_0"]
    W2 = model["arr_2"]
    W3 = model["arr_4"]

    b1 = model["arr_1"]
    b2 = model["arr_3"]
    b3 = model["arr_5"]

    return W1, b1, W2, b2, W3, b3


def _quantize(W, b):
    """
    Convert the float values into signed 8bit integers.
    The floats are multiplied by mult, and then rounded.
    The mult should be choosen so that the lowest and highest 
    values do not clip - avoid values outside [-128, 127]
    In case this is not followed, the values will overflow!

    NOTE: 256 choosen as it works for the current network.
    """
    mx = np.max([np.max(np.abs(W)), np.max(np.abs(b))])
    mult = np.floor(127 / mx) - 1

    W = np.round(W * mult)
    b = np.round(b * mult)

    assert np.all(W <  127)
    assert np.all(W > -128)
    assert np.all(b <  127)
    assert np.all(b > -128)

    return W.astype(np.int8), b.astype(np.int8)


def twos_complement(values):
    assert np.all(np.array(values) >= -128)
    assert np.all(np.array(values) <= 127)

    return np.array(values) & 0xff
    #TODO this is hacky, hardcoding 8 bits
    # return np.binary_repr(values, width=8)


def quantize_model(path):
    params = extract_model(path)

    quantized = []

    # TODO remove hardcoding
    for i in range(0, 6, 2):
        W, b = _quantize(params[i], params[i + 1])

        W = pad_bias(W, b).astype(np.int8).T
        W = twos_complement(W)
        quantized.append(W)

    return quantized


def pad_bias(matrix, bias):
    """ Pads the matrix with the bias, zeros, and one in the corner, to avoid having to compute the bias separately """
    bias = bias.reshape((1, len(bias)))

    side = np.zeros((matrix.shape[0] + 1, 1))
    side[-1] = 1

    matrix = np.concatenate((matrix, bias), axis=0)
    matrix = np.concatenate((matrix, side), axis=1)

    return matrix


def model_to_cisr_separate(path, channel_num):
    W1, W2, W3 = quantize_model(path)

    W1_cisr = encode_cisr_separate(W1, channel_num)
    W2_cisr = encode_cisr_separate(W2, channel_num)
    W3_cisr = encode_cisr_separate(W3, channel_num)

    return W1_cisr, W2_cisr, W3_cisr


def data2coe(path, data, bits_per_value=8):
    addresses = [0]
    output = []

    for d in data:
        addresses.append(addresses[-1] + len(d))
        output += d

    coe_gen.generate_coe(path, output, addresses, bits_per_value=bits_per_value)

    return output, addresses


def flip_around(W1_cisr, W2_cisr, W3_cisr, ch):
    """
    Takes matrices of 3 elements (val, col, len), of which each element is a list of length channel_num,
    and converts it to 3 elements (val, col, len), where the same channels are grouped together, not matrices
    """
    values  = W1_cisr[0] + W2_cisr[0] + W3_cisr[0]
    columns = W1_cisr[1] + W2_cisr[1] + W3_cisr[1]
    lengths = W1_cisr[2] + W2_cisr[2] + W3_cisr[2]

    indexing = [range(s, ch * 3, ch) for s in range(ch)]
    indexing = [item for sublist in indexing for item in sublist]  # flatten list - SO dark magic

    flipped_values  = [values[x]  for x in indexing]
    flipped_columns = [columns[x] for x in indexing]
    flipped_lengths = [lengths[x] for x in indexing]

    return flipped_values, flipped_columns, flipped_lengths
    

def model_to_coe(paths, channel_num):
    # each one of the values is a tuple (values, columns, lenghts), each of those values is a list of channels
    W1_cisr, W2_cisr, W3_cisr = model_to_cisr_separate(paths, channel_num)
    
    # values  = W1_cisr[0] + W2_cisr[0] + W3_cisr[0]
    # columns = W1_cisr[1] + W2_cisr[1] + W3_cisr[1]
    # lengths = W1_cisr[2] + W2_cisr[2] + W3_cisr[2]

    values, columns, lengths = flip_around(W1_cisr, W2_cisr, W3_cisr, channel_num)

    data2coe("values.coe",  values, bits_per_value=8)
    data2coe("columns.coe", columns, bits_per_value=16)
    data2coe("lengths.coe", lengths, bits_per_value=8)

    return values, lengths, columns


if __name__ == "__main__":
    import sys

    model_to_coe(sys.argv[1], int(sys.argv[2]))


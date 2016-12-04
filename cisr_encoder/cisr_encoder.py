import numpy as np


def generate_random_sparse(m, n, sparsity):
    values = np.random.rand(m, n)
    mask = np.random.binomial(1, 1 - sparsity, (m, n))

    return values * mask


def row_nnz(matrix, row_index):
    # gets the number of nonzero elements in the row
    return len(matrix[row_index].nonzero()[0])


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
                    while (row_nnz(matrix, next_free_row) == 0):
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


def extract_model(path):
    model = np.load(path)

    W1 = model["arr_0"]
    W2 = model["arr_2"]
    W3 = model["arr_4"]

    b1 = model["arr_1"]
    b2 = model["arr_3"]
    b3 = model["arr_5"]

    return W1, b1, W2, b2, W3, b3


def _quantize(values, mult=256):
    """
    Convert the float values into signed 8bit integers.
    The floats are multiplied by mult, and then rounded.
    The mult should be choosen so that the lowest and highest 
    values do not clip - avoid values outside [-128, 127]
    In case this is not followed, the values will overflow!

    NOTE: 256 choosen as it works for the current network.
    """
    rounded = np.round(values * mult)

    assert np.all(rounded <  127)
    assert np.all(rounded > -128)

    return rounded.astype(np.int8)


def quantize_model(path):
    params = extract_model(path)
    quantized = []

    for param in params:
        quantized.append(_quantize(param))

    return quantized


def model_to_cisr(path, channel_num):
    W1, b1, W2, b2, W3, b3 = quantize_model(path)

    W1_cisr = encode_cisr(W1, channel_num)
    W2_cisr = encode_cisr(W2, channel_num)
    W3_cisr = encode_cisr(W3, channel_num)

    return W1_cisr, b1, W2_cisr, b2, W3_cisr, b3
     

# def model_to_coe(path, channel_num):
    # W1_cisr, b1, W2_cisr, b2, W3_cisr, b3 = model_to_cisr(path, channel_num)

    

    


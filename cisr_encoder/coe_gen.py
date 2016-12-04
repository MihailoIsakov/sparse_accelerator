

def generate_coe(path, radix, memory, bytes_per_row=8):
    """
    Creates a COE file
    """
    f = open(path, 'w')

    f.write("memory_initialization_radix = " + str(radix) + "\n")
    f.write("memory_initialization_vector = \n")

    row_counter = 0
    for cell in memory:
        assert cell < 256
        f.write(format(cell, '02x') + " ")

        row_counter += 1
        if row_counter == bytes_per_row:
            f.write("\n")
            row_counter = 0






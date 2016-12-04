

def generate_coe(path, memory, radix=16, bytes_per_row=8):
    """
    Creates a COE file
    """
    f = open(path, 'w')

    f.write("; Storing NN weights and biases, along with images\n")
    f.write("; Stored sequentially as W1, b1, W2, b2, W3, b3, img1, img2, img3...\n")
    f.write("; To calculate the location of weights, biases or images, need to know their sizes!\n")
    f.write("; Memory stored as one byte values, written in hex format (radix = 16)\n")
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






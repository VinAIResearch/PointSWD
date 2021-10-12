class Writer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def write_arr_to_file(txtfile, meta, arr):
        writer = open(txtfile, "a")
        writer.write(meta)
        n, m = arr.shape
        for i in range(n):
            for j in range(m):
                writer.write(str(arr[i][j]))
                writer.write(" ")
            writer.write("\n")
        writer.close()
        return

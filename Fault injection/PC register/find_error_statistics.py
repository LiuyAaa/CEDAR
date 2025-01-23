import sys

class Statistics():
    def __init__(self, in_file, out_file):
        self.in_file = in_file
        self.out_file = out_file
        self.line_numbers = []

    def read_find_error(self):
        file = open(self.in_file)
        line = file.readline()
        round_count = 0
        flag_readed = False # Has the tag been read to the first line of this round
        while line:
            line_old = line  
            line = file.readline()
            # Counting laps
            if line_old[0:5] == "round":
                round_count += 1
                flag_readed = False
            # Number of error rows found during reading
            if line_old == "find control error!\n" and \
                 flag_readed == False:
                pre_string = "round %d : " % round_count
                self.line_numbers.append(pre_string + line)
                flag_readed = True 
        file.close()
        return self.line_numbers

    def write_line_number_file(self):
        file = open(self.out_file, 'w')
        file.writelines(self.line_numbers)
        file.close()

#st = Statistics(sys.argv[1], sys.argv[2])
st = Statistics("inject.output", "inject.find_error")
st.read_find_error()
st.write_line_number_file()


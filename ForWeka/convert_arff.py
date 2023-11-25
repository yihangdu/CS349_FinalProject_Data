import os
import sys



def parse_csv(fileName_in):

    file_in = open(fileName_in, "r")

    file_mat = []
    header = file_in.readline().strip().replace("/", "-").replace(" ", "_").split(",")

    label_idx = int(input("Give index of your label (starting from 0): \n"))
    while label_idx < 0 or label_idx >= len(header):
        label_idx = int(input("Give index of your label (starting from 0): \n"))


    header[label_idx], header[-1] = header[-1], header[label_idx]
    file_mat.append(header)
        

    for line in file_in:
        line = line.strip().replace("/", "-").split(",")

        line[label_idx], line[-1] = line[-1], line[label_idx] # put label at the last position
        
        file_mat.append(line)
    
    file_in.close()
    return file_mat

    
def category_process(file_mat, i):

    cate_class = set([file_mat[x][i] for x in range(1,len(file_mat))])

    cate_class_str = str(cate_class).replace("'", "")

    return f"@attribute {file_mat[0][i]} {cate_class_str}\n"  




def convert_arff(file_mat, fileName_out, relation):

    file_out = open(fileName_out, "w")
    file_out.write(f"@relation {relation}\n\n")
    
    i = 0
    while i < len(file_mat[0]):
        # Ctg stands for categorical, choose string instead if there is too many classes
        data_type = input(f"Define data type for {file_mat[0][i]} [N: Numeric][C: Ctg][D: Date][S: String]: ")

        data_type = data_type.upper()


        if data_type == "N":
            att = f"@attribute {file_mat[0][i]} numeric\n"

        elif data_type == "C":
            att = category_process(file_mat, i)

        elif data_type == "D":
            att = f'@attribute {file_mat[0][i]} date "yyyy-MM-dd"\n'

        elif data_type == "S":
            att = f"@attribute {file_mat[0][i]} string\n"

        else:
            continue

        i += 1

        file_out.write(att)

    file_out.write("\n@data\n")

    for i in range(1, len(file_mat)):
        write_line = str(file_mat[i]).replace(" ","").replace("[","").replace("]","").replace("'","")
        file_out.write(f'{write_line}\n')






if __name__ == "__main__":

    fileName_in = sys.argv[1]
    fileName_out = sys.argv[2]
    
    
    
    
    print("Parsing file ...")
    parsed = parse_csv(fileName_in)
    

    relation = input("Please provide @relation info: \n")
    convert_arff(parsed, fileName_out, relation)    

    
    
    
    


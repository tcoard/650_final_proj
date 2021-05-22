IN_FILE = "coala100.fa"
OUT_FILE = "esm100.fa"

### NOT CURRENTLY USED
def get_one_hot_encodings():
    # This is inefficient
    all_drugs = list()
    with open(IN_FILE, "r") as in_file:
        for line in in_file:
            if line.startswith(">"):
                splitted = line.split("|")
                if splitted[-1] not in all_drugs:
                    all_drugs.append(splitted[-1])

    drug_index = dict()
    for i, drug in enumerate(all_drugs):
        encoding = [0 for _ in range(len(all_drugs))]
        encoding[i] = 1
        drug_index[drug] = encoding
    return drug_index


def get_better_id(line, drug_index):
    splitted = line.split("|")
    drug = splitted[-1]
    drug = drug.replace("/", "_")
    return f'{splitted[0].split(" ")[0]}|{splitted[1].split(" ")[0]}|{drug}'


def get_rid_of_rare_aa(line):
    sequence = ""
    for aa in line.strip():
        # if not in the top 20 most frequent amino acids
        if aa not in "ACDEFGHIKLMNPQRSTVWY":
            aa = "X"
        sequence += aa
    return sequence

def main():
    # drug_index = get_one_hot_encodings()
    drug_index = None
    last_header = ""
    with open(IN_FILE, "r") as in_file, open(OUT_FILE, "w") as out_file:
        all_headers = set()
        for line in in_file:
            if line.startswith(">"):
                line = line.rstrip()
                last_header += get_better_id(line, drug_index)
            else:
                # I have seen messed up lines with headers and sequences in them
                # don't read them and just go to the next sequence
                if ">" not in line:
                    # make a unique identifyer based on the smaller id from get_better_id
                    if last_header in all_headers:
                        last_header = ">1_" + last_header[1:]
                        i = 2
                        while last_header in all_headers:
                            last_header[1] = f"{i}"
                            i += 1

                    # facebook's sequence length limit
                    if len(line) <= 1024:
                        all_headers.add(last_header)
                        out_file.write(last_header + "\n")
                        sequence = get_rid_of_rare_aa(line)
                        out_file.write(sequence + "\n")
                last_header = ""


if __name__ == "__main__":
    main()

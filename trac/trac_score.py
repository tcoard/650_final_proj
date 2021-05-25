import os
from Bio import SeqIO

RESULT_DIR = os.path.join(os.path.abspath(__file__), "..")
DATA_DIR = os.path.join(RESULT_DIR, "..", "data")
DATA_SUF = ".fa"

for filename in os.listdir(RESULT_DIR):
    total_count = 0
    correct_count = 0
    try:
        if 'fa_ab_res_predictions_results' in filename:
            data_name = filename.split(".")[0]
            dataset_file = open(os.path.join(DATA_DIR, data_name+DATA_SUF))
            dataset_records = SeqIO.parse(dataset_file, "fasta")
            result_file = open(os.path.join(RESULT_DIR,  filename))
            result_records = SeqIO.parse(result_file, "fasta")
            for data, result in zip(dataset_records, result_records):
                ground_truth_antibiotic = data.description.split("|")[-1]
                prediction_antibiotic = result.description.split("|")[-1]
                if ground_truth_antibiotic == prediction_antibiotic:
                    correct_count = correct_count + 1
                total_count = total_count + 1
            print("Accuracy for "+data_name+": "+str((correct_count*100.0)/total_count))
        else:
            with open(os.path.join('.', 'Final Project', 'Hamid-TRAC', filename)) as file:
                pass
    except PermissionError:
        pass
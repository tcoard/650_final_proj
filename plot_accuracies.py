# Creating an nicely-colored accuracy plot for our 10-folds

# CJ

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join('data', 'cross_validation_numbers.xlsx')
OUT_PATH = 'out'
MAX_FOLDS = 10
NAN_DELIM = "NAN"

# For conveniently packaging test data
class DataPacket():
    def __init__(self, dataset_name, model_designation):
        self.dataset = dataset_name
        self.model = model_designation
        self.data_dict = dict()

    def add_metric(self, fold_data, metric_label):
        self.data_dict[metric_label] = fold_data

    def get_metric_folds(self, metric_label):
        if not self.metric_available(metric_label):
            return []
        else:
            return self.data_dict[metric_label]

    def metric_available(self, metric_label):
        if metric_label in self.data_dict.keys():
            return True
        else:
            return False

# Read the formatted Excel document where the cross-validation data is held
def excel_parser(file_path):
    # Read in the Excel doc
    data_doc = pd.read_excel(DATA_PATH, usecols="A:M")

    # Get our columns and rows indices
    col_names = [key for key in data_doc.keys()]
    all_rows = [val[0] for val in np.argwhere(data_doc["Metrics"].notnull().values).tolist()]

    # Determine which rows begin separate data arrays and which datasets we have
    row_idxs = [val[0] for val in np.argwhere(data_doc["Dataset"].notnull().values).tolist()]
    datasets = np.unique(data_doc["Dataset"][row_idxs].values)
    models = np.unique(data_doc["Model Designation"][row_idxs].values)
    metrics = np.unique(data_doc["Metrics"].values)

    # Now replace null cells with NAN
    data_doc.fillna(NAN_DELIM, inplace=True)

    # Obtain numbers across all folds and package them all up for later plotting
    data = []
    last_packet_idx = -1
    for row in all_rows:
        details_row_idx = row_idxs[len(row_idxs) - np.argmax(np.array(row_idxs)[::-1] <= row) - 1]
        dataset_name = data_doc["Dataset"][details_row_idx]
        model_name = data_doc["Model Designation"][details_row_idx]
        metric_name = data_doc["Metrics"][row]
        fold_values = []
        for i in range(MAX_FOLDS):
            value = data_doc["Fold "+str(i+1)][row]
            if not value == NAN_DELIM:
                fold_values.append(value)
        if not last_packet_idx == details_row_idx:
            this_packet = DataPacket(dataset_name, model_name)
            data.append(this_packet)
        if fold_values:
            data[-1].add_metric(fold_values, metric_name)
        last_packet_idx = details_row_idx
    return [data, datasets.tolist(), models.tolist(), metrics.tolist()]

# Similar to the Friedberg paper, this function plots the average value of the specified metric with all datasets
def plot_metric(data_path, which_dataset, which_metric, save_plot=False):
    # Collect all of the data and pre-process it for plotting
    [validation_data, datasets_avail, models_avail, metrics_avail] = excel_parser(data_path)
    if not which_dataset in datasets_avail:
        print(which_dataset+": This dataset is not available for plotting.")
        print("Datasets available: "+str(datasets_avail))
        return
    if not which_metric in metrics_avail:
        print(which_metric+": This metric is not available for plotting.")
        print("Metrics available: "+str(metrics_avail))
        return
    # Extract only the datapackets whose model and metric match what is to be plotted
    applicable_packets = []
    model_names = []
    for dp in validation_data:
        if dp.dataset == which_dataset and dp.metric_available(which_metric):
            # If a particular dataset and model appears more than once, concatenate the fold data as if there were more folds
            formatted_model_name = dp.model.replace(" ", "\n")
            if formatted_model_name in model_names:
                for packet in applicable_packets:
                    if packet.model == formatted_model_name:
                        present_folds = packet.get_metric_folds(which_metric)
                        packet.add_metric(present_folds + dp.get_metric_folds(which_metric), which_metric)
                    else:
                        pass
            else:
                model_names.append(formatted_model_name)
                applicable_packets.append(dp)
    # One last pass to compute averages
    metric_averages = []
    for dp in applicable_packets:
        all_folds = dp.get_metric_folds(which_metric)
        metric_averages.append(sum(all_folds)/len(all_folds))
    
    # All of the Matplotlib Formatting
    cmap = plt.get_cmap('rainbow')
    y_plot_points = np.arange(len(model_names))
    plt.rcdefaults()
    fig, ax = plt.subplots()
    plt.grid(axis="x")

    ax.barh(y_plot_points, metric_averages, 
        color=cmap(np.linspace(0, 1, len(applicable_packets))), 
        height=0.3, align='center', zorder=2)

    ax.set_title(which_dataset+" Cross-Validated Average "+which_metric)

    ax.set_yticks(y_plot_points)
    ax.set_yticklabels(model_names)
    ax.invert_yaxis()
    ax.set_ylabel("Method", fontsize=15)

    ax.set_xlabel(which_metric, fontsize=15)
    ax.set_xlim([0, 1])
    ax.set_xticks(np.linspace(0, 1, 11))
    
    plt.tight_layout()
    # plt.show()

    if save_plot: 
        plt.savefig(os.path.join(OUT_PATH, which_dataset+"_"+which_metric+".png"))


if __name__ == '__main__':
    # A little misleading, I know, but it works!
    save_all_plots = False
    save_any_plots = False

    dataset_single = "Coala40"
    metric_single = "Accuracy"

    datasets = ["Coala40", "Coala70", "Coala100"]
    metrics = ["Accuracy", "False Positives", "False Negatives", "Recall", "Precision", "Specificity", "MCC (Matthews correlation coefficient", "F1 Score"]

    if save_all_plots:
        for dataset in datasets:
            for metric in metrics:
                plot_metric(DATA_PATH, dataset, metric, save_plot=save_any_plots)
    else:
        plot_metric(DATA_PATH, dataset_single, metric_single, save_plot=save_any_plots)
    
    
#!/common/home/rmartin/bin/python3
# This code writes scripts to do both (1) data preparation and (2) training/evaluation on video
# files for the ML behavior disriminator videos on the Rutgers CS computing infrastructure.
# (c) 2023 R. P. Martin. This code is licensed under the GNU General Public License (GPL), version 3
# This program takes the video list in a main data csv file and breaks it up into training and testing data sets to run k-fold validation.
# The main dataset.csv file is created by the command 'make_train_csv.py' script and is used as input to this script.
# After taking the input csv list, this script creates 4 additional kinds of files as output. What is produced is:
# 1. A set of N smaller dataset.csv files, used for cross-fold validation, broken up from the main dataset.csv file
# 2. N batch.sh shell scripts to to call the VidActRecDataprep.py script on the above data set files to perform the data preparation.
# 3. A global shell file (sbatch) to run the slurm sbatch command on the above batch files.
# 4. N training.sh shell scripts to to call the VidActRecTrain.py script on the above data tar files to perform training and evaluation.
# 5. A batch script that runs the training scripts using the slurm sbatch command and srun commands (to get GPUs)
# This script breaks up the large dataset file into multiple smaller randomized sets of 1/N size each
# The number of sets is controlled with the --k parameter below
# This script then creates these N dataset_XX.csv files from the main dataset file, as well
# as N shell file that calls the VidActRecDataprep.py script that creates the tar file for the training and testing script.
# The k-fold validation approach is described here:
# See: https://towardsdatascience.com/k-fold-cross-validation-explained-in-plain-english-659e33c0bc0
import argparse
import csv
import logging
import os
import random
import sys

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

parser = argparse.ArgumentParser(description="Create k-fold validation sets.")

parser.add_argument(
    "--datacsv",
    type=str,
    required=False,
    default="dataset.csv",
    help="name of the dataset, default dataset.csv",
)
parser.add_argument("--k",
                    type=int,
                    required=False,
                    default=3,
                    help="number of sets, default 3")
parser.add_argument(
    "--batchdir",
    type=str,
    required=False,
    default=".",
    help="working directory for the sbatch run jobs, default: .",
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default="01011970",
    help="Seed to use for randominizing the data sets, default: 01011970",
)
parser.add_argument(
    "--training",
    type=str,
    required=False,
    default="training-run",
    help="Name for the training script file, default: training-run",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    default="alexnet",
    help="Model to use for the training script, default: alexnet",
)
parser.add_argument(
    "--only_split",
    required=False,
    default=False,
    action="store_true",
    help="Set to finish after splitting the csv, default: False",
)
parser.add_argument(
    "--width",
    type=int,
    required=False,
    default=400,
    help=
    "Width of output images (obtained via cropping, after applying scale), default 400",
)
parser.add_argument(
    "--height",
    type=int,
    required=False,
    default=400,
    help=
    "Height of output images (obtained via cropping, after applying scale), default 400",
)
parser.add_argument(
    "--crop_x_offset",
    type=int,
    required=False,
    default=0,
    help=
    "The offset (in pixels) of the crop location on the original image in the x dimension, default 0",
)
parser.add_argument(
    "--crop_y_offset",
    type=int,
    required=False,
    default=0,
    help=
    "The offset (in pixels) of the crop location on the original image in the y dimension, default 0",
)
parser.add_argument(
    "--label_offset",
    required=False,
    default=0,
    type=int,
    help=
    'The starting value of classes when training with cls labels (the labels value is "cls"), default: 0',
)
parser.add_argument(
    "--training_only",
    type=bool,
    required=False,
    default=False,
    help="only generate the training set files, default: False",
)
parser.add_argument(
    "--path_to_file",
    type=str,
    required=False,
    default="bee_analysis",
    help="path to bee analysis files, default: bee_analysis",
)
parser.add_argument(
    "--frames_per_sample",
    type=int,
    required=False,
    default=1,
    help="Number of frames per sample, default 1",
)
parser.add_argument(
    "--epochs",
    type=int,
    required=False,
    default=10,
    help="Number of epochs to run, default 10",
)
parser.add_argument(
    "--gpus",
    required=False,
    default=1,
    type=int,
    help="Number of GPUs to use, default 1",
)

args = parser.parse_args()

# program_dir = "/research/projects/grail/rmartin/analysis-results/code/bee_analysis"
program_dir = os.path.join(os.getcwd(), args.path_to_file)
dataPrepProgram = os.path.join(program_dir, "VidActRecDataprep.py")
# The training python program
trainProgram = os.path.join(program_dir, "VidActRecTrain.py")  # ! FIX THIS TOO

# command to run the evaluation and training program
# trainCommand    = 'srun -G 1 python3 $TRAINPROGRAM --not_deterministic --epochs 10 --modeltype $MODEL --evaluate' # <eval-set> <a-set> <b-set> ...
# <eval-set> <a-set> <b-set> ...
trainCommand = f"python3 $TRAINPROGRAM --sample_frames {args.frames_per_sample} --not_deterministic --epochs {args.epochs} --modeltype $MODEL --label_offset $LABEL_OFFSET --evaluate"

datacsvname = args.datacsv
numOfSets = args.k
batchdir = args.batchdir
seed = args.seed
training_filename = args.training + ".sh"
model_name = args.model
width = args.width
height = args.height
crop_x_offset = args.crop_x_offset
crop_y_offset = args.crop_y_offset
label_offset = args.label_offset
training_only = args.training_only

logging.info(f"datset is {datacsvname}")

# set the random number generator
random.seed(seed)

with open(datacsvname) as datacsv:
    conf_reader = csv.reader(datacsv)
    header = next(conf_reader)
    # Remove all spaces from the header strings
    header = ["".join(col.split(" ")) for col in header]
    logging.info(f"header is {header}")
    file_col = header.index("filename")
    class_col = header.index("class")
    beginf_col = header.index("beginframe")
    endf_col = header.index("endframe")

    loop_counter = 0
    all_csv_rows = []
    # Put all the rows in the csv file into a list
    for row in conf_reader:
        all_csv_rows.append(row)
    pass

# create a randomized permutation
random_rows = random.shuffle(all_csv_rows)
numRows = len(all_csv_rows)

# figure out the number of files to put into each dataset
numFilesPerSet = int(numRows / numOfSets)
extraFiles = numRows % numOfSets

# create test_N and train_N files for each of the k folds
logging.info(
    f"Splitting {numRows} rows into {numFilesPerSet}/set with {extraFiles} extra"
)

# foreach dataset, construct a csv of the files in that set
baseNameFile = datacsvname.split(".csv")
baseName = baseNameFile[0]
setNum = 0
currentDir = os.getcwd()

# Write out the split csv files.
for dataset_num in range(numOfSets):
    dataset_filename = baseName + "_" + str(dataset_num) + ".csv"
    base_row = setNum * numFilesPerSet
    with open(dataset_filename, "w") as dsetFile:
        # write out the header row at the top of the set
        dsetFile.write("file, class, begin frame, end frame\n")
        # write out all the rows for this set
        for rowNum in range(base_row, base_row + numFilesPerSet):
            dsetFile.write(",".join(all_csv_rows[rowNum]))
            dsetFile.write("\n")
    setNum = setNum + 1

# Finish here if the only_split option was set.
if args.only_split:
    sys.exit(0)

if batchdir == ".":
    batchdir = currentDir

training_batch_file = open(training_filename, "w")
training_batch_file.write("#!/usr/bin/bash \n")
training_batch_file.write(
    "# batch file for getting the training results \n \n")
training_batch_file.write("cd " + currentDir + " \n")
training_batch_file.write(
    "echo start-is: `date` \n \n")  # add start timestamp to training file

trainCommand = trainCommand.replace("$MODEL", model_name)

for dataset_num in range(numOfSets):
    train_job_filename = "train" + "_" + str(dataset_num) + ".sh"

    # open the batch file that runs the testing and training commands
    with open(train_job_filename, "w") as trainFile:
        trainFile.write("#!/usr/bin/bash \n")
        # trainFile.write("#SBATCH --gpus-per-node=1 \n")
        trainFile.write("# command to run \n \n")
        trainFile.write("export TRAINPROGRAM=" + trainProgram + "\n")
        trainFile.write("cd " + currentDir + " \n")
        trainFile.write("echo start-is: `date` \n \n")  # add start timestamp
        traincommand_local = trainCommand.replace("$TRAINPROGRAM",
                                                  trainProgram)
        traincommand_local = traincommand_local.replace(
            "$LABEL_OFFSET", str(label_offset))
        traincommand_local = (traincommand_local + " " + baseName + "_" +
                              str(dataset_num) + ".tar")
        for trainingSetNum in range(numOfSets):
            if int(trainingSetNum) != int(dataset_num):
                traincommand_local = (traincommand_local + " " + baseName +
                                      "_" + str(trainingSetNum) + ".tar")

        trainFile.write(
            traincommand_local +
            "\n")  # write the training command to the training command
        trainFile.write("echo end-is: `date` \n \n")  # add end timestamp
        training_batch_file.write(f"sbatch -G {args.gpus} -o " + baseName +
                                  "_trainlog_" + str(dataset_num) + ".log " +
                                  train_job_filename +
                                  " \n")  # add end timestamp to training file

    setNum = setNum + 1

training_batch_file.write(
    "echo end-is: `date` \n \n")  # add end timestamp to training file
training_batch_file.close()

logging.info("Done writing dataset and job files")
# change the permissions of the shell scripts to be executable.
os.system("chmod 777 *")

import dist_mat as dm
import csv
import gen_data as gd

dist_mat = dm.one_frame(gd.read_csv_and_create_tuples('data.csv'))

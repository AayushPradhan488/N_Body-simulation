import random
import csv

def generate_random_points_csv(num_points, filename='random_points.csv'):
    points = [(int(random.uniform(-1000, 1000)), int(random.uniform(-1000, 1000)), 1, 0, 0) for _ in range(num_points)]
    
    # Save points to a CSV file
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['X', 'Y', 'm'])  # Header
        csv_writer.writerows(points)

def read_csv_and_create_tuples(filename='random_points.csv'):
    points_list = []

    # Read points from CSV file
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            x, y, m, vx, vy = map(int, row)
            points_list.append((x, y))

    return points_list

# Generate 1000 random points with integer coordinates and save to CSV
#generate_random_points_csv(10000,'data.csv')

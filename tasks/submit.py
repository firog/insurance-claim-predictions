import csv
import pandas as pd

def generate_submission(ids, predictions):
    with open('submission.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'target'])
        zipped = zip(ids, predictions[:, 1])
        for id_, prediction in zipped:
            writer.writerow([id_, prediction])
            
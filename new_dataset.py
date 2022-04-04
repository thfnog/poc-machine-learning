import csv

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

header = ['userId','locationId','rating','timestamp']
rows = [
    {70, 20, 2.5, 835355697},
    {70, 34, 4, 835355697},
    {70, 46, 5, 835355697},
    {70, 53,4.5, 835355697},
    {70, 58,4.5, 835355697}
]

def createNewData():
    with open('datas/new_data.csv', 'w') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)
# import statements
import csv
import json
import os
from scipy import stats
import numpy as np
# constants
# input file to convert into csv
folder = './data/raw/'
file_name = 'decahouse_polls_2020-10'
input_file = folder + file_name + '.txt'
output_file = f'./data/query-data/{file_name}.csv'
chunk = 0
chunk_size = 100000
count = 0
num_failed = 2 # number of jsons that failed to parse. For now, we manually change this value to restart the script.

try:
    with open(output_file) as f:
        reader = csv.reader(f)
        count = sum(1 for _ in reader)
    chunk = count // chunk_size
    # print('Count', count)
    # print('Chunk size', chunk_size)
    # print('Chunk:', chunk)
except:
    chunk = 0

# to save memory, we read in line by line and work in chunks
# first, we want to get a sense of how many lines there are so we need to open and read the file line by line
# while doing this, we also calculate the offset needed, which chunk * chunk_size, since that tells us which line
# to start reading from
with open(input_file, 'r') as file_stream:
    num_lines = 0
    offset = 0
    restart_offset = 0
    for line in file_stream:
        num_lines += 1
        offset += len(line)
        if num_lines == chunk * chunk_size + num_failed:
            restart_offset = offset
    file_stream.seek(restart_offset)
    print('Number of jsons:', num_lines)
    print('Number of chunks:', num_lines // chunk_size)
    print('Number of chunks processed:', chunk)
    print('Number of files processed:', count - 1)
    if count - 1 + num_failed >= num_lines:
        print('Finished processing all files')
        exit(0)
    data_jsons = []
    chunk_end_index = min(num_lines, chunk_size)
    i = 0
    for line in file_stream:
        if i == chunk_end_index:
            break
        try:
            json_line = json.loads(line)
        except:
            num_failed += 1
            continue
        data_jsons.append(json_line)
        i += 1
    print('Number of jsons failed to parse:', num_failed)
    csv_fields = ['tweet_id', 'text', 'options', 'relevant']
    csv_rows = []
    for tweet in data_jsons:
        tweet_id = str(tweet['id'])
        text =  tweet['text']
        # print(json.dumps(tweet, indent=2, sort_keys=True))
        # print('............')
        options = [opt['text'] for opt in tweet["entities"]["polls"][0]["options"]]

        lower_options = [opt.lower() for opt in options]
        rel_options = ['biden', 'trump', 'clinton']
        found_options = [False, False, False]
        for opt in lower_options:
            if 'biden' in opt:
                found_options[0] = True
            elif 'trump' in opt:
                found_options[1] = True
            elif 'clinton' in opt:
                found_options[2] = True
        rel_label = 1 if found_options[0] and found_options[1] or found_options[1] and found_options[2] else 0
        cur_row = [tweet_id, text, options, rel_label]
        csv_rows.append(cur_row)


# writing to csv file  
with open(output_file, 'a+') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    
    if os.stat(output_file).st_size == 0:
        # writing the fields  
        csvwriter.writerow(csv_fields)  
        
    # writing the data rows  
    csvwriter.writerows(csv_rows) 


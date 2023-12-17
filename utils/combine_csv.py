import pandas as pd

def combine_csvs(csv_list, output_file):
    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()

    # Loop through each CSV file in the list
    for csv_file in csv_list:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Append the DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

    print(f"CSV files combined successfully. Output saved to {output_file}")

# Example usage:
common_path = './data/query-data/'
# csv_files = [common_path + f'decahouse_polls_2020-10_cleanedV2-{i}.csv' for i in range(10)]
# output_csv = common_path + 'decahouse_polls_2020-10_cleanedV2.csv'

csv_files = [common_path + f'decahouse_polls_2020-10_cleanedV2.csv', 
             common_path + f'vote-query-stephen - vote-query-stephen_cleanedV2.csv']

output_csv = common_path + 'decahouse_2020-10_stephen_combined.csv'

combine_csvs(csv_files, output_csv)
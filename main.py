from helpful_functions import *

df = load_data("data.csv")

df = find_and_delete_nulls(df)

show_basic_stats(df)

df = create_totalordervalue_and_averageordervalue(df)

prepare_and_execute_clustering(df)
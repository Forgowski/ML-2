from helpful_functions import *

df = load_data("data.csv")

df = find_and_delete_nulls(df)

find_nagative_quantity(df)

show_basic_stats(df)

df = create_totalordervalue_and_averageordervalue(df)

basic_df = df

df = prepare_and_execute_clustering(df)

df = tsne(df)

group_with_the_biggest_amount_of_returns(df)

df = new_df_for_rfm(basic_df)

random_forest_rfm(df)
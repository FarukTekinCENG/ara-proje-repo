# .env file:
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=

# if working with db: init postgres db with the tables 
!chmod +x table_def/bootstrap.sh
!./table_def/bootstrap.sh

# if working with db: split dataset train - test within db
python -m scripts.split_pool --fraction 0.2 --seed 42 --yes

# if working with csv: prepare balanced dataset
python -m scripts.prepare_balanced_dataset --force_download --target_per_class 500 --log_append

# plot graph: all test data blocks
python scripts/plot_graph.py --input results/results.xlsx --output_dir graphs --accuracy_percent

# plot graph: for specific test data block
python scripts/plot_graph.py --input results/results.xlsx --output_dir graphs --accuracy_percent --test_index 3

# plot graph: as if all file is only one test data block
python scripts/plot_graph.py --input results/results.xlsx --output_dir graphs --accuracy_percent --no_split
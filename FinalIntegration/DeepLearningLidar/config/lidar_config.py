import argparse

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Config for the Implementation')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
import argparse
import toml

def delete_groups(input_file, output_file, groups):
    """
    Deletes specified groups from a pyproject.toml file.

    Args:
        input_file (str): Path to the input pyproject.toml file.
        output_file (str): Path to the output pyproject.toml file.
        groups (list of str): List of group names to delete.

    Returns:
        None
    """
    # Load the file
    data = toml.load(input_file)

    # Delete the specified groups
    for group in groups:
        group_key = f'group.{group}'
        if group_key in data['tool']['poetry']:
            del data['tool']['poetry'][group_key]

    # Write the file back
    with open(output_file, 'w') as f:
        toml.dump(data, f)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Delete groups from a pyproject.toml file.')
    parser.add_argument('--groups', nargs='+', default=['dev', 'finetune'],
                        help='Names of the groups to delete. Default is ["dev", "finetune"].')
    parser.add_argument('--input', default='./train/pyproject.toml',
                        help='Path to the input pyproject.toml file. Default is "./train/pyproject.toml".')
    parser.add_argument('--output', default='./train/pyproject-core.toml',
                        help='Path to the output pyproject.toml file. Default is "./train/pyproject.toml".')

    # Parse arguments
    args = parser.parse_args()

    # Delete groups
    delete_groups(args.input, args.output, args.groups)

if __name__ == '__main__':
    main()


def read_blocks4procs(path_to_file):

    with open(f'{path_to_file}/blocks4procs.inp') as f:
        lines = [i.split(',') for i in f.readlines() if not i.strip().startswith('#')]
        blocks4procs = []
        for line in lines:
            blocks4procs.append([int(i) for i in line])

    return blocks4procs

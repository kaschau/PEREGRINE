
def read_blocks4procs(config):

    with open(f"{config['io']['inputdir']}/blocks4procs.inp") as f:
        lines = [i.split(',') for i in f.readlines() if not i.strip().startswith('#')]
        blocks4procs = []
        for line in lines:
            blocks4procs.append([int(i) for i in line])

    return blocks4procs

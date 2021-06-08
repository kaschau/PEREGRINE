from ..files import config_file

def read_config_file(file_path):

    config = config_file()

    with open(file_path,'r') as f:
        for line in [i for i in f.readlines() if not i.replace(' ','').startswith('#') and i.strip() != '']:
            if '=' not in line.strip():
                header = line.strip()
                try:
                    section = config[header]
                    continue
                except KeyError:
                    raise ValueError('Section header {} not part of standard config file, exiting.'.format(header))

            nocomment = line.strip().split('#')[0]
            key,val = tuple(nocomment.replace(' ','').split('='))
            try: #convert numbers to floats or ints
                if '.' in val:
                    config[header][key] = float(val)
                else:
                    config[header][key] = int(val)
            except ValueError:
                if val in ['True','true','t','T']:
                    config[header][key] = True
                elif val in ['False','true','f','F']:
                    config[header][key] = False
                else:
                    config[header][key] = val

    return config

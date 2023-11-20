prefix = '/home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/'


def extract(tokens, option):
    if option in tokens:
        i = tokens.index(option)
        tok = tokens[i+1]
        tok = tok.replace(prefix, '')
        return [tok]
    else:
        return []


def extract_startswith(tokens, prefix):
    all_tokens = []
    for tok in tokens:
        if tok.startswith(prefix):
            all_tokens.append(tok)
    return all_tokens


def extract_endswith(tokens, suffix):
    all_tokens = []
    for tok in tokens:
        if tok.endswith(suffix):
            all_tokens.append(tok)
    return all_tokens


def common_prefix(m):
    if not m: return []
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return [s1[:i] + '*']
    return [s1 + '*']


with open('build-stage1.log') as fh:
    for line in fh:
        line = line.strip()
        tokens = line.split()

        src = extract(tokens, '-c')
        dst = extract(tokens, '-o')

        slibs = extract_startswith(tokens, '-l')
        dlibs = extract_endswith(tokens, '.so')
        libs = list(set(slibs + dlibs) - set(dst))

        objs = extract_endswith(tokens, '.o')
        objs_dir = common_prefix(objs)

        if not dst or dst[0].endswith('.o') or 'bin/' in dst[0]:
            pass
        elif dst[0] in objs_dir:
            print(src + libs, '==>', dst)
        else:
            print(src + objs_dir + libs, '==>', dst)

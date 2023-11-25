import os
import subprocess
prefix = '/home/tk/Desktop/nvme0n1/pytorch-that-I-successfully-built/'


def extract(tokens, option, preserve=False):
    results = []
    for i, tok in enumerate(tokens):
        if tok == option:
            tok = tokens[i+1]
            tok = tok.replace(prefix, '')
            if preserve:
                results.append(option)
            results.append(tok)
    return results


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


def print_simple_lines():
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


def expand_headers(src_file='tensor_new.cpp', options=['-H', '-c']):
    with open('build-stage1.log') as fh:
        for line in fh:
            line = line.strip()
            tokens = line.split()
            src = extract(tokens, '-c')
            if src and src[0].endswith(src_file):
                isystem = extract(tokens, '-isystem', True)
                includes = extract_startswith(tokens, '-I')
                cc = tokens[1]
                src = src[0]
                cmd = [cc] + options + [src] + includes + isystem
                output = subprocess.check_output(' '.join(cmd), shell=True, stderr=subprocess.STDOUT)
                return output.decode().strip()


def expand_headers_struct(src_file='tensor_new.cpp', dst_file=None):
    output = expand_headers(src_file)
    path = []
    for line in output.split('\n'):
        if len(line.split()) > 2:
            break
        level, header = line.split()
        level, header = len(level) - 1, header.replace(prefix, '')
        while level < len(path):
            path.pop()
        path.append(header)
        print(path)


def gen_source_ctags(src_file='tensor_new.cpp'):
    output = expand_headers(src_file, options=['-M'])
    with open('tags.list', 'w') as fh:
        for line in output.split('\n'):
            fields = line.split(':')
            line = fields[-1]
            line = line.strip('\\')
            line = line.strip()
            fh.write(line + '\n')
    os.system('ctags -L tags.list --c++-kinds=+p --fields=+iaS --extras=+q')


if __name__ == '__main__':
    import fire
    fire.Fire({
        'print_simple_lines': print_simple_lines,
        'expand_headers': expand_headers,
        'expand_headers_struct': expand_headers_struct,
        'gen_source_ctags': gen_source_ctags,
    })

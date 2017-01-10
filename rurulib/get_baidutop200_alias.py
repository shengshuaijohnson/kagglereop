

file1 = 'alias_by_table.txt'
file2 = 'baidu_top_200.txt'

top200 = []
with open(file2, 'r') as f:
    for line in f:
        if '.' in line:
            line = line[:-2]
        top200.append(line.strip())

with open(file1, 'r') as f:
    for line in f:
        for top in top200:
            if line.startswith(top) and not '&' in line:
                with open('top_200_alias.txt','a') as out:
                    # top200.remove(top)
                    out.write(line.strip() + '\n')

for i in top200:
    print i

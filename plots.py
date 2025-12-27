import matplotlib.pyplot as plt
import re
import numpy as np

plt.rcParams.update({'font.size': 14})

with open('output.txt', 'r') as f:
    raw_text = f.read()

clean_text = re.sub(r'\'', '', raw_text)
sections = re.split(r'EXPERIMENT \d+:', clean_text)

# PARSING EXPERIMENT 1
exp1_names = []
exp1_hll_err = []
exp1_rec_err = []
exp1_pattern = re.compile(r'([\w\.-]+\(?[ \w\.]*\)?)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')

for line in sections[1].split('\n'):
    line = line.strip()
    if not line or '---' in line or 'Dataset' in line:
        continue
    match = exp1_pattern.search(line)
    if match:
        name = match.group(1).strip()
        name = name.replace('.txt', '')
        if name.startswith('Synthetic'):
            name = 'Synthetic'
        exp1_names.append(name)
        exp1_hll_err.append(float(match.group(4)))
        exp1_rec_err.append(float(match.group(6)))

# PARSING EXPERIMENT 2
exp2_data = {}
file_blocks = re.split(r'Analyzing Memory Impact for:', sections[2])
for block in file_blocks[1:]:
    lines = block.strip().split('\n')
    filename = lines[0].split('(')[0].strip()
    exp2_data[filename] = {'hll': {'m': [], 'obs': [], 'theor': []}, 
                           'rec': {'k': [], 'obs': [], 'theor': []}}
    
    current_algo = None
    for line in lines:
        if '[HyperLogLog]' in line: current_algo = 'hll'
        elif '[Recordinality]' in line: current_algo = 'rec'
        
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5 and parts[0].isdigit() and current_algo == 'hll':
            exp2_data[filename]['hll']['m'].append(int(parts[1]))
            exp2_data[filename]['hll']['obs'].append(float(parts[3]))
            exp2_data[filename]['hll']['theor'].append(float(parts[4]))
        elif len(parts) >= 4 and parts[0].isdigit() and current_algo == 'rec':
            exp2_data[filename]['rec']['k'].append(int(parts[0]))
            exp2_data[filename]['rec']['obs'].append(float(parts[2]))
            exp2_data[filename]['rec']['theor'].append(float(parts[3]))

# PARSING EXPERIMENT 3
exp3_alpha = []
exp3_hll_err = []
exp3_rec_err = []

for line in sections[3].split('\n'):
    parts = line.split()
    if len(parts) >= 5 and re.match(r'^\d', parts[0]):
        exp3_alpha.append(float(parts[0]))
        exp3_hll_err.append(float(parts[2]))
        exp3_rec_err.append(float(parts[4]))

# GENERATING THE PLOTS
# Experiment 1: Bar Plot
plt.figure(figsize=(12, 6))
x_idx = np.arange(len(exp1_names))
width = 0.35
plt.gca().set_axisbelow(True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.bar(x_idx - width/2, exp1_hll_err, width, label='HyperLogLog', color='skyblue')
plt.bar(x_idx + width/2, exp1_rec_err, width, label='Recordinality', color='salmon')
plt.xticks(x_idx, exp1_names, rotation=45, ha='right')
plt.ylabel('Relative Error (%)')
plt.legend()
plt.tight_layout()
plt.savefig('exp1_barplot.png')

# Experiment 2: Line Plots for each analyzed file
for fname, data in exp2_data.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # HLL Line Plot
    ax1.plot(data['hll']['m'], data['hll']['obs'], 'o-', label='Observed', color='blue')
    ax1.plot(data['hll']['m'], data['hll']['theor'], 's--', label='Theoretical', color='cyan')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('m')
    ax1.set_ylabel('Standard Error (%)')
    ax1.set_title('HyperLogLog')
    ax1.legend(); ax1.grid(True)
    
    # REC Line Plot
    ax2.plot(data['rec']['k'], data['rec']['obs'], 'o-', label='Observed', color='red')
    ax2.plot(data['rec']['k'], data['rec']['theor'], 's--', label='Theoretical', color='orange')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('k')
    ax2.set_ylabel('Standard Error (%)')
    ax2.set_title('Recordinality')
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'exp2_lineplot_{fname.replace(".", "_")}.png')

# Experiment 3: Alpha Line Plot
plt.figure(figsize=(8, 5))
plt.plot(exp3_alpha, exp3_hll_err, 'o-', label='HyperLogLog', color='blue')
plt.plot(exp3_alpha, exp3_rec_err, 's-', label='Recordinality', color='red')
plt.xlabel('α')
plt.ylabel('Relative Error (%)')
plt.ylim(top = 10)
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('exp3_alphaplot.png')
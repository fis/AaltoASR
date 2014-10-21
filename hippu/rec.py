#!/usr/bin/python

import time
import string
import sys
import os
import re

if len(sys.argv) < 7 or len(sys.argv) > 8:
    sys.stderr.write('rec.py customized for aaltoasr-rec internal use only\n')
    sys.exit(1)

# Set your decoder swig path in here!
sys.path.append(sys.argv[1] + "/lib/site-packages");

import Decoder


def runto(frame):
    while (frame <= 0 or t.frame() < frame):
        if (not t.run()):
            break

def rec(start, end):
    st = os.times()
    t.reset(start)
    t.set_end(end)
    runto(0)
    et = os.times()
    duration = et[0] + et[1] - st[0] - st[1] # User + system time
    frames = t.frame() - start;
    sys.stdout.write('DUR: %.2fs (Real-time factor: %.2f)\n' %
                     (duration, duration * 125 / frames))

##################################################
# Initialize
#

rootdir = sys.argv[1]
model = sys.argv[2]
hmms = model+".ph"
dur = model+".dur"
lexicon = rootdir + "/model/morph19k.lex"
ngram = rootdir + "/model/morph19k_D20E10_varigram.bin"
lookahead_ngram = rootdir + "/model/morph19k_2gram.bin"
recipefile = sys.argv[3]
lna_path = sys.argv[4]
lm_scale = int(sys.argv[5])
global_beam = 250
num_batches = 1 #int(sys.argv[4])
batch_index = 1 #int(sys.argv[5])

do_phoneseg = int(sys.argv[6])
morphseg_file = sys.argv[7] if len(sys.argv) == 8 else None

##################################################


##################################################
# Load the recipe
#
f=open(recipefile,'r')
recipelines = f.readlines()
f.close()

# Extract the lna files

if num_batches <= 1:
    target_lines = len(recipelines)
else:
    target_lines = int(len(recipelines)/num_batches)
if target_lines < 1:
    target_lines = 1

cur_index = 1
cur_line = 0

lnafiles=[]
for line in recipelines:
    if num_batches > 1 and cur_index < num_batches:
        if cur_line >= target_lines:
            cur_index += 1
            if (cur_index > batch_index):
                break
            cur_line -= target_lines

    if num_batches <= 1 or cur_index == batch_index:
        result = re.search(r"lna=(\S+)", line)
        if result:
            lnafiles = lnafiles + [result.expand(r"\1")]
    cur_line += 1

# Check LNA path
if lna_path[-1] != '/':
    lna_path = lna_path + '/'

##################################################
# Recognize
#

sys.stderr.write("loading models\n")
t = Decoder.Toolbox(0, hmms, dur)

t.set_optional_short_silence(1)

t.set_cross_word_triphones(1)

t.set_require_sentence_end(1)

t.set_verbose(1)
t.set_print_text_result(1)
t.set_print_state_segmentation(do_phoneseg)
t.set_lm_lookahead(1)

word_end_beam = int(2*global_beam/3);
trans_scale = 1
dur_scale = 3

t.set_duration_scale(dur_scale)
t.set_transition_scale(trans_scale)
t.set_lm_scale(lm_scale)

t.set_word_boundary("<w>")

sys.stderr.write("loading lexicon\n")
try:
    t.lex_read(lexicon)
except:
    print "phone:", t.lex_phone()
    sys.exit(-1)
t.set_sentence_boundary("<s>", "</s>")

sys.stderr.write("loading ngram\n")
t.ngram_read(ngram, 1)
t.read_lookahead_ngram(lookahead_ngram)

t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth

t.set_global_beam(global_beam)
t.set_word_end_beam(word_end_beam)
t.set_token_limit(30000)
t.set_prune_similar(3)

#t.set_print_probs(0)
#t.set_print_indices(0)
#t.set_print_frames(0)

print "BEAM: ", global_beam
print "WORD_END_BEAM: ", word_end_beam
print "LMSCALE: ", lm_scale
print "DURSCALE: ", dur_scale

if morphseg_file is not None:
    t.set_generate_word_graph(1)

for lnafile in lnafiles:
    t.lna_open(lna_path + lnafile, 1024)

    print "LNA:", lnafile
    print "REC: ",
    rec(0,-1)

    if morphseg_file is not None:
        t.write_word_history(morphseg_file)

# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
import select
import shutil
import struct
import sys
import tempfile
import textwrap
from itertools import groupby
from os.path import basename, join
import subprocess
from subprocess import call, check_output

"""Aalto ASR tools for CSC Hippu environment.

This module contains the required glue code for using the Aalto ASR
tools for simple speech recognition and forced alignment tasks, in the
CSC Hippu environment.
"""

# Overall settings and paths to data files

rootdir = '/work/t40511_research/htkallas/aaltoasr'
models = {
    '16k': { 'path': 'speecon_mfcc_gain3500_occ225_1.11.2007_20', 'srate': 16000, 'fstep': 128, 'default': 1 },
    '8k': { 'path': 'speechdat_mfcc_gain4000_occ350_13.2.2008_20', 'srate': 8000, 'fstep': 128 }
    }

def bin(prog):
    return join(rootdir, 'bin', prog)

default_args = {
    'model': [m for m, s in models.items() if 'default' in s][0],
    'lmscale': 30,
    'align-window': 1000,
    'align-beam': 100.0,
    'align-sbeam': 100,
    }

# Command-line help

help = {
    'rec': {
        'desc': 'Recognize speech from an audio file.',
        'usage': '%(prog)s [options] input [input ...]',
        'modes': ('trans', 'segword', 'segmorph', 'segphone'),
        'defmode': 'trans',
        'extra': ('The MODE parameter specifies which results to include in the generated output.  '
                  'It has the form of a comma-separated list of the terms "trans", "segword", '
                  '"segmorph" and "segphone", denoting a transcript of the recognized text, '
                  'a word-level, statistical-morpheme-level or phoneme-level segmentation, '
                  'respectively. The listed items will be included in the plaintext output. '
                  'The default is "trans". For more details, see the User\'s Guide at: '
                  'http://users.ics.aalto.fi/htkallas/guide.html')
        },
    'align': {
        'desc': 'Align a transcription to a speech audio file.',
        'usage': '%(prog)s [options] -t transcript [-t transcript ...] input [input ...]',
        'modes': ('segword', 'segphone'),
        'defmode': 'segword',
        'extra': ('The MODE parameter specifies which results to include in the generated output. '
                  'It has the form of a comma-separated list of the terms "segword" and "segphone", '
                  'denoting a word-level or phoneme-level segmentation, respectively. The listed '
                  'items will be included in the plaintext output. The default is "segword".  '
                  'For more details, see the User\'s Guide at: '
                  'http://users.ics.aalto.fi/htkallas/guide.html')
        }
    }

class ModelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == 'list':
            sys.stderr.write('supported models:\n')
            for m in sorted(models.keys()):
                sys.stderr.write('  %s [sample rate: %d Hz]\n' % (m, models[m]['srate']))
            sys.exit(2)
        setattr(namespace, self.dest, values)

# Class for implementing the rec/align/adapt tools

class AaltoASR(object):
    def __init__(self, tool, args=None):
        """Initialize and parse command line attributes."""

        self.tool = tool

        # Ask argparse to grok the command line arguments

        thelp = help[tool]

        parser = argparse.ArgumentParser(description=thelp['desc'], usage=thelp['usage'], epilog=thelp['extra'])

        parser.add_argument('input', help='input audio file', nargs='+')
        parser.add_argument('-t', '--trans', help='provide an input transcript file', metavar='file',
                            required=True if tool == 'align' else False, action='append')
        parser.add_argument('-a', '--adapt', help='provide a speaker adaptation file', metavar='file',
                            default=None)
        parser.add_argument('-o', '--output', help='output results to file [default stdout]', metavar='file',
                            type=argparse.FileType('w'), default=sys.stdout)
        parser.add_argument('-m', '--mode', help='which kind of results to output (see below)', metavar='MODE',
                            default=thelp['defmode'])
        parser.add_argument('-T', '--tg', help='output also a Praat TextGrid segmentation to file', metavar='file',
                            type=argparse.FileType('w'), default=None)
        if tool == 'rec':
            parser.add_argument('-r', '--raw', help='produce raw recognizer output (with morph breaks)',
                                action='store_true')
            parser.add_argument('-s', '--split',
                                help='split input file to segments of about S seconds [default %(const)s if present]',
                                metavar='S', nargs='?', type=float, default=None, const=60.0)
        parser.add_argument('-n', '--cores', help='run tasks simultaneously on up to N cores [default 1]',
                            metavar='N', type=int, default=1)
        parser.add_argument('-v', '--verbose', help='print output also from invoked commands', action='store_true')
        parser.add_argument('-q', '--quiet', help='do not print status messages', action='store_true')
        parser.add_argument('--tempdir', help='directory for temporary files', metavar='D')

        params = parser.add_argument_group('speech recognition parameters')
        params.add_argument('-M', '--model', help='acoustic model to use; "-M list" for list [default "%(default)s"]',
                            metavar='M', default=default_args['model'], choices=['list']+list(models.keys()),
                            action=ModelAction)
        if tool == 'rec':
            params.add_argument('-L', '--lmscale', help='language model scale factor [default %(default)s]', metavar='L',
                                type=int, default=default_args['lmscale'])
        if tool == 'align':
            params.add_argument('--align-window',
                                help='set the Viterbi window for alignment [default %(default)s]', metavar='W',
                                type=int, default=default_args['align-window'])
            params.add_argument('--align-beam', help='set alignment log-probability beam [default %(default)s]', metavar='B',
                                type=float, default=default_args['align-beam'])
            params.add_argument('--align-sbeam', help='set alignment state beam [default %(default)s]', metavar='S',
                                type=int, default=default_args['align-sbeam'])
        params.add_argument('--noexp', help='disable input transcript expansion', action='store_true')

        parser.add_argument('--keep', help=argparse.SUPPRESS, action='store_true')

        self.args = parser.parse_args(args)

        # Check applicable arguments for validity

        for infile in self.args.input:
            if not os.access(infile, os.R_OK):
                err('input file not readable: {0}'.format(infile), exit=2)

        self.transfiles = None
        if self.args.trans:
            self.transfiles = [open(f, 'rb') for f in self.args.trans]
            if len(self.transfiles) != len(self.args.input):
                err('number of transcript files does not match number of inputs', exit=2)

        self.mode = set()
        for word in self.args.mode.split(','):
            if word not in thelp['modes']:
                err('invalid output mode: %s; valid: %s' % (word, ', '.join(thelp['modes'])), exit=2)
            self.mode.add(word)

        self.tg = self.args.tg is not None

        self.model = models[self.args.model]
        self.mpath = join(rootdir, 'model', self.model['path'])
        self.mcfg = self.mpath + ('.adapt.cfg' if self.args.adapt else '.cfg')
        self.margs = ['-b', self.mpath, '-c', self.mcfg]

        if self.args.adapt is not None:
            self.margs.extend(('-S', self.args.adapt))

        self.cores = self.args.cores


    def __enter__(self):
        """Make a working directory for a single execution."""

        self.workdir = tempfile.mkdtemp(prefix='aaltoasr', dir=self.args.tempdir)

        return self

    def __exit__(self, type, value, traceback):
        """Clean up the working directory and any temporary files."""

        if self.workdir.find('aaltoasr') >= 0 and not self.args.keep: # sanity check
            shutil.rmtree(self.workdir)


    def convert_input(self):
        """Convert input audio to something suitable for the model."""

        self.audiofiles = []

        for idx, infile in enumerate(self.args.input):
            fileid = 'input-{0}'.format(idx)
            base = join(self.workdir, fileid)

            finfo = { 'id': fileid,
                      'path': infile }

            if self.tool == 'rec' and self.args.split is not None:
                self.log('splitting input file {0} to {1}-second segments'.format(infile, self.args.split))
                finfo['files'] = split_audio(self.args.split, infile, base, self.model)
            else:
                self.log('converting input file {0} to {1} Hz mono'.format(infile, self.model['srate']))

                audiofile = base + '.wav'

                if call([bin('sox'), infile,
                         '-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                         audiofile]) != 0:
                    err("input conversion of '%s' with sox failed" % self.args.input, exit=1)

                finfo['files'] = [{ 'start': 0, 'file': audiofile }]

            self.audiofiles.append(finfo)

        nfiles = sum(len(finfo['files']) for finfo in self.audiofiles)

        if self.cores > nfiles:
            self.cores = nfiles
            self.log('using only {0} core{1}; no more audio segments'.format(
                    self.cores, '' if self.cores == 1 else 's'))


    def align(self):
        """Do segmentation with the Viterbi alignment tool."""

        self.convert_input()
        #audiofile = self.audiofiles[0]['file']

        self.log('computing Viterbi alignment')

        recipe = join(self.workdir, 'align.recipe')
        alignfiles = [join(self.workdir, '{0}.phn'.format(f['id'])) for f in self.audiofiles]
        outfiles = [join(self.workdir, '{0}.align'.format(f['id'])) for f in self.audiofiles]

        self.phones = [None] * len(self.audiofiles)

        for fidx, finfo in enumerate(self.audiofiles):
            audiofile = finfo['files'][0]['file'] # never split when aligning

            # Convert the transcription to a phoneme list

            phones = text2phn(self.transfiles[fidx], self.workdir, expand=not self.args.noexp)
            self.phones[fidx] = phones

            # Write out the cross-word triphone transcript

            with open(alignfiles[fidx], 'w', encoding='iso-8859-1') as f:
                for pnum, para in enumerate(phones):
                    phns = para['phns']
                    f.write('__\n')
                    for phnum, ph in enumerate(phns):
                        if ph == '_':
                            f.write('_ #{0}:{1}\n'.format(pnum, phnum))
                        else:
                            prevph, nextph = '_', '_'
                            for prev in range(phnum-1, -1, -1):
                                if phns[prev] != '_': prevph = phns[prev]; break
                            for next in range(phnum+1, len(phns)):
                                if phns[next] != '_': nextph = phns[next]; break
                            f.write('{0}-{1}+{2} #{3}:{4}\n'.format(prevph, ph, nextph, pnum, phnum))
                f.write('__\n')

        # Make a recipe for the alignment

        with open(recipe, 'w') as f:
            for fidx, finfo in enumerate(self.audiofiles):
                f.write('audio={0} transcript={1} alignment={2} speaker=UNK\n'.format(
                        finfo['files'][0]['file'], alignfiles[fidx], outfiles[fidx]))

        # Run the Viterbi alignment

        cmd = [bin('align'),
               '-r', recipe, '-i', '1',
               '--swins', str(self.args.align_window),
               '--beam', str(self.args.align_beam),
               '--sbeam', str(self.args.align_sbeam)]
        cmd.extend(self.margs)
        self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))

        self.alignments = outfiles


    def phone_probs(self):
        """Create a LNA file for the provided audio."""

        self.convert_input()

        self.log('computing acoustic model likelihoods')

        recipe = join(self.workdir, 'input.recipe')
        lnafiles = []

        # Construct an input recipe

        with open(recipe, 'w') as f:
            for fidx, finfo in enumerate(self.audiofiles):
                lnas = []
                for i, audiofile in enumerate(finfo['files']):
                    lnafile = join(self.workdir, '{0}-{1}.lna'.format(finfo['id'], i))
                    lnas.append(lnafile)
                    f.write('audio={0} lna={1} speaker=UNK\n'.format(audiofile['file'], lnafile))
                lnafiles.append(lnas)

        # Run phone_probs on the files

        cmd = [bin('phone_probs'),
               '-r', recipe, '-C', self.mpath+'.gcl', '-i', '1',
               '--eval-ming', '0.2']
        cmd.extend(self.margs)
        self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))

        self.lna = lnafiles


    def rec(self):
        """Run the recognizer for the generated LNA file."""

        self.log('recognizing speech')

        # Call rec.py on the lna files

        lnamap = {}
        histmap = {}

        recipe = join(self.workdir, 'rec.recipe')
        with open(recipe, 'w') as f:
            for fidx, lnafiles in enumerate(self.lna):
                for i, lnafile in enumerate(lnafiles):
                    lna_id = basename(lnafile)
                    lnamap[lna_id] = (fidx, i)
                    histmap[(fidx, i)] = len(histmap)
                    f.write('lna={0}\n'.format(lna_id))

        cmd = [bin('rec.py'),
               rootdir, self.mpath, recipe, self.workdir, str(self.args.lmscale),
               '1' if 'segphone' in self.mode or self.tg else '0',
               join(self.workdir, 'wordhist') if 'segmorph' in self.mode or 'segword' in self.mode or self.tg else '']
        rec_out = self.run(cmd, output=True, batchargs=lambda i, n: (str(n), str(i)))
        rec_out = ''.join(o.decode('iso-8859-1') for o in rec_out)

        # Parse the recognizer output to extract recognition result and state segmentation

        rec_file, rec_filepart = -1, -1
        rec_start = 0
        rec_trans = [[[] for i in lnafiles] for lnafiles in self.lna]
        rec_seg = [[[] for i in lnafiles] for lnafiles in self.lna]

        re_lna = re.compile(r'^LNA: (.*)$')
        re_trans = re.compile(r'^REC: (.*)$')
        re_seg = re.compile(r'^(\d+) (\d+) (\d+)$')

        for line in rec_out.splitlines():
            m = re_lna.match(line)
            if m is not None:
                rec_file, rec_filepart = lnamap[m.group(1)]
                rec_start = self.audiofiles[rec_file]['files'][rec_filepart]['start']
                continue
            m = re_trans.match(line)
            if m is not None:
                rec_trans[rec_file][rec_filepart].append(m.group(1))
                continue
            m = re_seg.match(line)
            if m is not None:
                start, end, state = m.group(1, 2, 3)
                rec_seg[rec_file][rec_filepart].append((rec_start+int(start), rec_start+int(end), int(state)))
                continue

        rec_trans = [[i for l in translist for i in l] for translist in rec_trans]
        rec_seg = [[i for l in seglist for i in l] for seglist in rec_seg]

        if not all(trans for trans in rec_trans):
            sys.stderr.write(rec_out)
            err('unable to find recognition transcript in output', exit=1)

        self.rec = [' <w> '.join(trans).strip() for trans in rec_trans]

        fstep = self.model['fstep']

        # If necessary, find labels for states and write an alignment file

        if 'segphone' in self.mode or self.tg:
            labels = get_labels(self.mpath + '.ph')

            self.alignments = []

            for fidx, finfo in enumerate(self.audiofiles):
                alignment = join(self.workdir, '{0}.align'.format(finfo['id']))
                with open(alignment, 'w', encoding='iso-8859-1') as f:
                    for start, end, state in rec_seg[fidx]:
                        f.write('%d %d %s\n' % (start*fstep, end*fstep, labels[state]))
                self.alignments.append(alignment)

        # If necessary, parse the generated word history file

        if 'segmorph' in self.mode or 'segword' in self.mode or self.tg:
            re_line = re.compile(r'^(\S+)\s+(\d+)')
            self.morphsegs = []

            for fidx, finfo in enumerate(self.audiofiles):
                morphseg = []

                for i, audiofile in enumerate(finfo['files']):
                    seg = []
                    file_start = audiofile['start']
                    prev_end = file_start

                    with open(join(self.workdir, 'wordhist-{0}'.format(histmap[(fidx,i)])), 'r', encoding='iso-8859-1') as f:
                        for line in f:
                            m = re_line.match(line)
                            if m is None: continue # skip malformed
                            morph, end = m.group(1), file_start + int(m.group(2))
                            seg.append((prev_end*fstep, end*fstep, morph))
                            prev_end = end

                    while len(seg) > 0 and (seg[0][2] == '<s>' or seg[0][2] == '<w>'):
                        seg.pop(0)
                    while len(seg) > 0 and (seg[-1][2] == '</s>' or seg[-1][2] == '<w>'):
                        seg.pop()

                    if len(seg) == 0:
                        continue
                    if len(morphseg) > 0:
                        morphseg.append((morphseg[-1][1], seg[0][0], '<w>'))
                    morphseg.extend(seg)

                self.morphsegs.append(morphseg)


    def gen_output(self):
        """Construct the requested outputs."""

        out = self.args.output
        out_started = [False]

        def hdr(text, suffix='\n\n'):
            if out_started[0]: out.write('\n\n')
            out.write('### {0}{1}'.format(text, suffix))
            out_started[0] = True

        for fidx, finfo in enumerate(self.audiofiles):
            if len(self.audiofiles) > 1:
                hdr('Input file: {0}'.format(finfo['path']), suffix='\n')
            self.gen_output_one(fidx, hdr)

    def gen_output_one(self, fidx, hdr):
        """Construct the requested outputs for one file."""

        # Start by parsing the state alignment into a phoneme segmentation

        if 'segphone' in self.mode or ('segword' in self.mode and self.tool == 'align') or self.tg:
            # Parse the raw state-level alignment

            rawseg = []

            re_line = re.compile(r'^(\d+) (\d+) ([^\.]+)\.(\d+)(?: #(\d+):(\d+))?')
            with open(self.alignments[fidx], 'r', encoding='iso-8859-1') as f:
                for line in f:
                    m = re_line.match(line)
                    if m is None:
                        err('invalid alignment line: %s' % line, exit=1)
                    phpos = (int(m.group(5)), int(m.group(6))) if m.group(5) else None
                    rawseg.append((int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), phpos))

            # Recover the phoneme level segments from the state level alignment file

            phseg = []

            cur_ph, cur_state = None, 0

            for start, end, rawph, state, phpos in rawseg:
                ph = trip2ph(rawph)

                if ph == cur_ph and state == cur_state + 1:
                    # Hack: try to fix cases where the first state of a phoneme after a long silence
                    # actually includes (most of) the silence too.  Ditto for last states.
                    if rawph.startswith('_-') and state == 1 and start - phseg[-1][0] > 2*(end-start):
                        phseg[-1][0] = start
                    if rawph.endswith('+_') and state == 2 and end - start > 2*(start - phseg[-1][0]):
                        continue # don't update end
                    phseg[-1][1] = end
                else:
                    phseg.append([start, end, ph, phpos])

                cur_ph, cur_state = ph, state

            # Split into list of utterances

            uttseg = []

            for issep, group in groupby(phseg, lambda item: item[2] == '__'):
                if not issep:
                    uttseg.append(list(group))

        # Merge phoneme segmentation to words in transcript if aligning

        if ('segword' in self.mode or self.tg) and self.tool == 'align':
            warned = False

            phonepos = dict((pos, (start, end, ph))
                            for start, end, ph, pos
                            in (i for utt in uttseg for i in utt))
            wordseg = []

            at = 0
            for uttidx, utt in enumerate(self.phones[fidx]):
                phstack = []

                for phidx, ph in enumerate(utt['phns']):
                    if (uttidx, phidx) in phonepos:
                        start, end, aph = phonepos[(uttidx, phidx)]
                        if aph != ph:
                            err('segmenter confused: phoneme mismatch at {0}/{1}: {2} != {3}'.format(uttidx, phidx, aph, ph), exit=1)
                        at = end
                    else:
                        if not warned:
                            self.log('warning: gaps in aligned output (check transcript?)')
                            warned = True
                        start, end = at, at
                    phstack.append((start, end, ph))

                wseg = []

                for w in intersperse(self.phones[fidx][uttidx]['words'], '_'):
                    start, end = 0, 0
                    for phnum, ph in enumerate(w.replace('-', '_')):
                        if phnum == 0: start = phstack[0][0]
                        end = phstack[0][1]
                        phstack.pop(0)
                    if w != '_': wseg.append((start, end, w))

                wordseg.append(wseg)

        # Merge morpheme segmentation into words if required

        if ('segword' in self.mode or self.tg) and self.tool == 'rec':
            wordseg = []

            for issep, group in groupby(self.morphsegs[fidx], lambda item: item[2] == '<s>' or item[2] == '</s>'):
                if issep: continue

                utt = []

                for issep, wordgroup in groupby(group, lambda item: item[2] == '<w>'):
                    if issep: continue
                    word = list(wordgroup)
                    utt.append((word[0][0], word[-1][1], ''.join(m[2] for m in word)))

                wordseg.append(utt)

        # Generate requested plaintext outputs

        out = self.args.output
        srate = float(self.model['srate'])

        if 'trans' in self.mode:
            hdr('Recognizer transcript:')
            if not self.args.raw:
                for utt in self.rec[fidx].split('<s>'):
                    utt = utt.replace('</s>', '')
                    utt = utt.replace(' ', '')
                    utt = utt.replace('<w>', ' ').strip()
                    if len(utt) > 0:
                        out.write('%s\n' % utt)
            else:
                out.write('%s\n' % self.rec[fidx])

        if 'trans' in self.mode and self.args.trans:
            hdr('Recognition accuracy:')
            phones = text2phn(self.transfiles[fidx], self.workdir, expand=not self.args.noexp)
            sclite(out, self.rec[fidx], phones, self.workdir)

        if 'segword' in self.mode:
            hdr('Word-level segmentation:')
            for utt in intersperse(wordseg, ()):
                if len(utt) == 0: out.write('\n')
                else:
                    for start, end, w in utt:
                        out.write('%.3f %.3f %s\n' % (start/srate, end/srate, w))

        if 'segmorph' in self.mode:
            hdr('Morpheme[*]-level segmentation:   ([*] statistical)')
            for start, end, morph in self.morphseg:
                if morph == '<s>' or morph == '</s>': continue
                elif morph == '<w>': out.write('\n')
                else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, morph))

        if 'segphone' in self.mode:
            hdr('Phoneme-level segmentation:')
            for utt in intersperse(uttseg, ()):
                if len(utt) == 0: out.write('\n\n')
                else:
                    for start, end, ph in utt:
                        if ph == '_': out.write('\n')
                        else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, ph))

        # Generate Praat TextGrid output

        if self.tg:
            tgfile = self.args.tg

            tgfile.write(('''
File type = "ooTextFile"
Object class = "TextGrid"

xmin = %.3f
xmax = %.3f
tiers? <exists>
size = %d
item []:
''' % (min(uttseg[0][0][0], wordseg[0][0][0]) / srate,
       max(uttseg[-1][-1][1], wordseg[-1][-1][1]) / srate,
       4 if self.tool == 'rec' else 3)).lstrip())

            tierno = 1

            # Utterance tier

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "utterance"\n')
            tgfile.write('        xmin = %.3f\n' % (wordseg[0][0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (wordseg[-1][-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % len(wordseg))

            for uttnum, utt in enumerate(wordseg):
                tgfile.write('        intervals [%d]:\n' % (uttnum+1))
                tgfile.write('            xmin = %.3f\n' % (utt[0][0] / srate))
                tgfile.write('            xmax = %.3f\n' % (utt[-1][1] / srate))
                tgfile.write('            text = "%s"\n' % ' '.join(w[2] for w in utt))

            # Word tier

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "word"\n')
            tgfile.write('        xmin = %.3f\n' % (wordseg[0][0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (wordseg[-1][-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % sum(len(utt) for utt in wordseg))

            for wnum, word in enumerate(w for utt in wordseg for w in utt):
                tgfile.write('        intervals [%d]:\n' % (wnum+1))
                tgfile.write('            xmin = %.3f\n' % (word[0] / srate))
                tgfile.write('            xmax = %.3f\n' % (word[1] / srate))
                tgfile.write('            text = "%s"\n' % word[2])

            # Morph tier

            if self.tool == 'rec':
                fmorphseg = [morph for morph in self.morphseg if morph[2][0] != '<']

                tgfile.write('    item[%d]:\n' % tierno); tierno += 1
                tgfile.write('        class = "IntervalTier"\n')
                tgfile.write('        name = "morph"\n')
                tgfile.write('        xmin = %.3f\n' % (fmorphseg[0][0] / srate))
                tgfile.write('        xmax = %.3f\n' % (fmorphseg[-1][1] / srate))
                tgfile.write('        intervals: size = %d\n' % len(fmorphseg))

                for morphnum, morph in enumerate(fmorphseg):
                    tgfile.write('        intervals [%d]:\n' % (morphnum+1))
                    tgfile.write('            xmin = %.3f\n' % (morph[0] / srate))
                    tgfile.write('            xmax = %.3f\n' % (morph[1] / srate))
                    tgfile.write('            text = "%s"\n' % morph[2])

            # Phoneme tier

            fphseg = [ph for ph in phseg if ph[2][0] != '_']

            tgfile.write('    item[%d]:\n' % tierno); tierno += 1
            tgfile.write('        class = "IntervalTier"\n')
            tgfile.write('        name = "phone"\n')
            tgfile.write('        xmin = %.3f\n' % (fphseg[0][0] / srate))
            tgfile.write('        xmax = %.3f\n' % (fphseg[-1][1] / srate))
            tgfile.write('        intervals: size = %d\n' % len(fphseg))

            for phnum, ph in enumerate(fphseg):
                tgfile.write('        intervals [%d]:\n' % (phnum+1))
                tgfile.write('            xmin = %.3f\n' % (ph[0] / srate))
                tgfile.write('            xmax = %.3f\n' % (ph[1] / srate))
                tgfile.write('            text = "%s"\n' % ph[2])


    def adapt(self, output):
        """Generate a CMLLR adaptation transform from aligned output."""

        if any(len(f['files']) != 1 for f in self.audiofiles):
            err('impossible: adaptation with splitting enabled', exit=1)

        self.log('cleaning up aligned transcriptions')

        for a in self.alignments:
            alignment_fixup(a, a+'.fixup')

        self.log('training CMLLR adaptation matrix')

        recipe = join(self.workdir, 'adapt.recipe')
        with open(recipe, 'w') as f:
            for fidx, finfo in enumerate(self.audiofiles):
                f.write('audio={0} alignment={1}.fixup speaker=UNK\n'.format(finfo['files'][0]['file'], self.alignments[fidx]))

        spk = join(self.workdir, 'adapt.spk')
        with open(spk, 'w') as f:
            f.write('\n'.join(['speaker UNK', '{', 'feature cmllr', '{', '}', '}', '']))

        cmd_out = sys.stderr if self.args.verbose else open(os.devnull, 'w')

        if call([bin('mllr'),
                 '-b', self.mpath, '-c', self.mpath+'.adapt.cfg',
                 '-M', 'cmllr', '-O', '-i', '1',
                 '-r', recipe, '-S', spk,
                 '-o', output], stdout=cmd_out, stderr=cmd_out) != 0:
            err('mllr failed', exit=1)


    def run(self, cmdline, batchargs=None, output=False):
        if self.args.verbose:
            cmd_out = sys.stderr
            self.log('run: {0}'.format(' '.join(cmdline)))
        else:
            cmd_out = open(os.devnull, 'w')

        if batchargs is None or self.cores == 1:
            cmds = (cmdline,)
        else:
            cmds = [tuple(cmdline) + tuple(batchargs(i, self.cores))
                    for i in range(1, self.cores+1)]

        procs = []

        for cmd in cmds:
            try:
                p = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE if output else cmd_out,
                                     stderr=cmd_out)
            except Exception as e:
                err('command "{0}" failed: {1}'.format(' '.join(cmd), e), exit=1)

            procs.append(p)

        if output:
            fds = [p.stdout.fileno() for p in procs]
            fdmap = dict((fd, i) for i, fd in enumerate(fds))
            outputs = [bytearray() for i in range(len(cmds))]
            while fds:
                readable = select.select(fds, (), ())[0]
                for fd in readable:
                    out = os.read(fd, 4096)
                    if len(out) == 0:
                        fds.remove(fd)
                    else:
                        outputs[fdmap[fd]].extend(out)

        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0:
                err('command "{0}" failed: non-zero return code: {1}'.format(
                        ' '.join(cmds[i]), p.returncode), exit=1)

        if output:
            return [bytes(out) for out in outputs]


    def log(self, msg):
        if not self.args.quiet:
            sys.stderr.write('%s: %s\n' % (basename(sys.argv[0]), msg))

# "Free-format" transcript to phoneme list conversion

def text2phn(input, workdir, expand=True):
    """Convert a transcription to a list of phonemes."""

    # Read in, split to trimmed paragraphs

    if 'read' in dir(input):
        input = input.read()

    if type(input) is bytes:
        try: input = input.decode('utf-8')
        except UnicodeError:
            input = input.decode('iso-8859-1')

    if type(input) is not str:
        err('unable to understand input: {0}'.format(repr(input)), exit=1)

    input = filter(None, (re.sub(r'\s+', ' ', para.strip()) for para in input.split('\n\n')))

    # Attempt to expand abbreviations etc., if requested

    if expand:
        expander = join(rootdir, 'lavennin', 'bin', 'lavennin')
        exp_in = join(workdir, 'expander_in.txt')
        exp_out = join(workdir, 'expander_out.txt')

        with open(exp_in, 'w', encoding='iso-8859-1') as f:
            f.write('\n'.join(input) + '\n')

        if call([expander, workdir, exp_in, exp_out]) != 0:
            err('transcript expansion script failed', exit=1)

        with open(exp_out, 'r', encoding='iso-8859-1') as f:
            input = filter(None, (para.strip() for para in f.readlines()))

    # Go from utterances to list of words

    input = [re.sub(r'\s+', ' ',
                    re.sub('[^a-zåäö -]', '', para.lower())
                    ).strip().split()
             for para in input]

    # Add phoneme lists

    return [{ 'words': para, 'phns': list('_'.join(para).replace('-', '_')) } for para in input]

# Fixup for discontinuity-caused unexpected states in alignment files

def alignment_fixup(infile, outfile):
    re_line = re.compile(r'^(\d+)( \d+ [^\.]+\.(\d+).*)$')

    with open(infile, 'r', encoding='iso-8859-1') as fi, open(outfile, 'w', encoding='iso-8859-1') as fo:
        pstate = -2
        pstart = None

        for line in fi:
            line = line.rstrip('\n')
            m = re_line.match(line)
            if m is None:
                err('invalid alignment line: %s' % line, exit=1)
            start, rest, state = m.group(1), m.group(2), int(m.group(3))

            if pstate == -2 or state == 0 or state == pstate+1:
                if pstart is None:
                    fo.write(line + '\n')
                else:
                    fo.write(pstart + rest + '\n')
                pstate, pstart = state, None
            else:
                pstate = -1
                if pstart is None: pstart = start

# SCLITE recognition transcript scoring

def sclite(out, rec, phones, workdir):
    reffile = join(workdir, 'sclite.ref')
    hypfile = join(workdir, 'sclite.hyp')

    hyptext = ' '.join(word for para in phones for word in para['words'])
    hyptext = hyptext.replace('-', ' ')

    with open(reffile, 'w', encoding='iso-8859-1') as f:
        f.write(hyptext + ' (spk-0)\n')
    with open(reffile+'.c', 'w', encoding='iso-8859-1') as f:
        f.write(hyptext.replace(' ', '_') + ' (spk-0)\n')

    rectext = rec.replace(' ', '')
    rectext = re.sub(r'</?[sw]>', ' ', rectext)
    rectext = re.sub(r'\s+', ' ', rectext).strip()

    with open(hypfile, 'w', encoding='iso-8859-1') as f:
        f.write(rectext + ' (spk-0)\n')
    with open(hypfile+'.c', 'w', encoding='iso-8859-1') as f:
        f.write(rectext.replace(' ', '_') + ' (spk-0)\n')

    for mode, suffix, flags in (('Letter', '.c', ['-c']), ('Word', '', [])):
        out.write('{0} error report:\n'.format(mode))

        cmd = [bin('sclite')]
        cmd.extend(flags)
        cmd.extend(['-r', reffile+suffix, '-h', hypfile+suffix,
                    '-s', '-i', 'rm', '-o', 'sum', 'stdout'])
        scout = check_output(cmd).decode('ascii', errors='ignore')

        for line in scout.split('\n'):
            if line.find('SPKR') >= 0 or line.find('Sum/Avg') >= 0:
                out.write(line + '\n')

# Miscellaneous helpers

def err(msg, exit=-1):
    sys.stderr.write('%s: error: %s\n' % (basename(sys.argv[0]), msg))
    if exit >= 0: sys.exit(exit)

def intersperse(iterable, delim):
    """Haskell intersperse: return elements of iterable interspersed with delim."""

    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delim
        yield x

def trip2ph(triphone):
    if len(triphone) == 5: return triphone[2]
    elif triphone == '__': return '__'
    elif triphone[0] == '_': return '_'

    err('unknown triphone: %s' % triphone, exit=1)

def get_labels(phfile):
    re_index = re.compile(r'^\d+ (\d+) (.*)')

    labels = {}

    with open(phfile, 'r', encoding='iso-8859-1') as ph:
        if ph.readline() != 'PHONE\n': err('bad phoneme file: wrong header', exit=1)

        phcount = int(ph.readline())

        while True:
            line = ph.readline().rstrip()
            if not line: break
            line = line

            m = re_index.match(line)
            if m is None:
                err('bad phoneme file: wrong index line: %s' % line, exit=1)

            phnstates = int(m.group(1))
            phname = m.group(2)

            line = ph.readline().strip()
            phstates = [int(s) for s in line.split()]
            if len(phstates) != phnstates:
                err('bad phoneme file: wrong number of states', exit=1)

            s = 0
            for idx in phstates:
                ph.readline() # skip the transition probs
                if idx < 0: continue # dummy states don't need labels
                labels[idx] = '%s.%d' % (phname, s)
                s += 1

    return labels

def split_audio(seglen, infile, basepath, model):
    """Split an input audio file to approximately seglen-second segments,
    at more or less silent positions if possible.  Frame size used when
    splitting will match the frame size of the model, and the output list
    gives start offsets of the segments in terms of that.
    """

    # compute target segment length in frames

    srate = model['srate']
    framesize = model['fstep']
    segframes = int(seglen * srate / framesize)
    max_offset = segframes / 5

    # generate frame energy mapping for the input audio

    raw_energy = []

    with subprocess.Popen([bin('sox'), infile,
                           '-t', 'raw', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1',
                           '-'],
                          stdout=subprocess.PIPE) as sox:
        while True:
            frame = sox.stdout.read(2*framesize)
            if len(frame) < 2*framesize:
                break
            frame = struct.unpack('={0}h'.format(framesize), frame)

            mean = float(sum(frame)) / len(frame)
            raw_energy.append(math.sqrt(sum((s-mean)**2 for s in frame)))

    if not raw_energy:
        err("input conversion of '{0}' with sox resulted in no frames".format(infile), exit=1)

    # moving-average smoothing for the energy

    energy = [0.0]*len(raw_energy)

    for i in range(len(energy)):
        wnd = raw_energy[max(i-10,0):i+11]
        energy[i] = sum(wnd)/len(wnd)

    # determine splitting positions

    segments = []
    at = 0

    while at < len(energy):
        left = len(energy) - at

        if left <= 1.5 * segframes:
            take = left
        else:
            target = at + segframes
            minpos = max(0, int(target - max_offset))
            maxpos = min(len(energy), int(target + max_offset + 1))
            pos = minpos + min(enumerate(energy[minpos:maxpos]),
                               key=lambda v: (1+abs(minpos+v[0]-target)/max_offset)*v[1])[0]
            take = pos - at

        segments.append((at, at+take))
        at += take

    # generate the resulting audio files

    audiofiles = []

    for i, (start, end) in enumerate(segments):
        starts = start*framesize
        lens = (end-start)*framesize

        audiofile = '{0}-{1}.wav'.format(basepath, i)

        if call([bin('sox'), infile,
                 '-t', 'wav', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1',
                 audiofile,
                 'trim', str(starts)+'s', str(lens)+'s']) != 0:
            err("input conversion of '{0}' (frames {1}-{2}) with sox failed".format(infile, start, end), exit=1)

        audiofiles.append({ 'start': start, 'file': audiofile })

    return audiofiles

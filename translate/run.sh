#!/bin/bash

set -e

OPT=(-b 1000)

python mt.py    ${OPT[@]}               2>/dev/null | awk '/done/{ print; exit }'
python mt.py -g ${OPT[@]}               2>/dev/null | awk '/done/{ print; exit }'
python mt.py    ${OPT[@]} --nstep       2>/dev/null | awk '/done/{ print; exit }'
python mt.py -g ${OPT[@]} --nstep       2>/dev/null | awk '/done/{ print; exit }'
python mt.py    ${OPT[@]} --nstep --pad 2>/dev/null | awk '/done/{ print; exit }'
python mt.py -g ${OPT[@]} --nstep --pad 2>/dev/null | awk '/done/{ print; exit }'




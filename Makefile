# BPAC Paper 1 -- top-level pipeline driver.
#
# Targets map onto the repo's Parts, which map onto the spec's phases:
#   Phase 0 (T1-T2)   -- this scaffolding + common/
#   Phase 1 (T3-T5)   -- data integrity        -> lands inside Part_A/B/C + new scripts
#   Phase 2 (T6-T8)   -- corrected numbers      -> lands inside Part_A/B/C
#   Phase 3 (T9-T10)  -- confound defence       -> lands inside Part_C
#   Phase 4 (T11-T19) -- phonetic accent map    -> Part_D
#   Phase 5 (T20)     -- validation             -> Part_E
#   Phase 6 (T21+)    -- release                -> paper/, release/
#
# Until each task is implemented, `phaseN` targets run whatever already
# exists for that phase (each Part's run_all.sh skips missing analyze.py
# scripts rather than failing).

SHELL := /bin/bash

.PHONY: all part_a part_b part_c part_d part_e \
        phase1 phase2 phase3 phase4 phase5 phase6 \
        check clean

part_a:
	cd Part_A && ./run_all.sh

part_b:
	cd Part_B && ./run_all.sh

part_c:
	cd Part_C && ./run_all.sh

part_d:
	cd Part_D && ./run_all.sh

part_e:
	cd Part_E && ./run_all.sh

# Phase aliases -- see header comment for the phase -> Part mapping.
phase1: part_a part_b part_c
phase2: part_a part_b
phase3: part_c
phase4: part_d
phase5: part_e
phase6:
	@echo "Phase 6 (release, T21+) has no automated target yet -- see release/README.md."

all: part_a part_b part_c part_d part_e

# T8 count-consistency audit -- reconciles every reports/summary.json
# against CORPUS_FROZEN.json. Not yet implemented (lands with T8).
check:
	@if [ -f common/consistency_check.py ]; then \
		python common/consistency_check.py; \
	else \
		echo "common/consistency_check.py not yet implemented (T8)."; \
	fi

clean:
	find . -type d -name reports -not -path "./paper/*" -exec rm -rf {} +
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Removed all reports/ and __pycache__ directories. data/ and data_clean/ untouched."

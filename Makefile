# Squish developer convenience targets.
#
# Coverage mirrors the CI `coverage` job: on Apple Silicon (CI=1 disables the
# sandbox MLX-import guard in tests/conftest.py) the MLX inference paths execute,
# giving a representative number. The Metal-SIGABRT-at-collection files are
# excluded — same list as the CI `test` job.

PYTEST ?= python -m pytest
COV_IGNORES := \
	--ignore=tests/test_git_integration.py \
	--ignore=tests/quant/test_int4_loader.py \
	--ignore=tests/test_sqint2_linear.py \
	--ignore=tests/test_backend_unit.py \
	--ignore=tests/quant/test_int3_linear_unit.py \
	--ignore=tests/hardware/test_fused_kernels_unit.py

.PHONY: test coverage coverage-html cov-open

## test: run the full test suite
test:
	$(PYTEST) tests/ -q

## coverage: run the suite with coverage and print a terminal report (term-missing)
coverage:
	CI=1 $(PYTEST) tests/ $(COV_IGNORES) \
		--cov=squish --cov-report=term-missing --cov-report=xml --timeout=120 -q || true

## coverage-html: same as `coverage` but also writes an htmlcov/ report
coverage-html:
	CI=1 $(PYTEST) tests/ $(COV_IGNORES) \
		--cov=squish --cov-report=term:skip-covered --cov-report=html --timeout=120 -q || true
	@echo "HTML report: htmlcov/index.html"

## cov-open: open the HTML coverage report (macOS)
cov-open:
	@open htmlcov/index.html 2>/dev/null || echo "Run 'make coverage-html' first"
